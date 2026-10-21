//! Client side of the pipeline: holds the token embedding and the output head, so token ids and
//! logits never leave the user's machine, and drives generation through a [`Pipeline`].

use std::{
    net::{IpAddr, Ipv4Addr, SocketAddr},
    sync::Arc,
    time::{Duration, Instant},
};

use afo_codec::CodecKind;
use afo_engine::{KvCache, Shard};
use afo_proto::{Forward, HopStat, Inbound, Msg, Net, NetOpts};
use anyhow::{Context, Result, bail};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use tokio::{sync::mpsc::UnboundedReceiver, time::timeout};

/// Everything between the embedding and the output head.
#[allow(async_fn_in_trait)]
pub trait Pipeline {
    /// Runs the hidden states of `n_tokens` tokens at positions `pos..` through all layers.
    async fn forward(&mut self, pos: usize, hidden: &[f32], n_tokens: usize) -> Result<(Vec<f32>, Vec<HopStat>)>;
    async fn close(&mut self) -> Result<()>;
}

/// All layers on this machine; the reference that split execution is checked against.
pub struct LocalPipeline {
    shard: Shard,
    cache: KvCache,
}

impl LocalPipeline {
    pub fn new(shard: Shard) -> Self {
        let cache = shard.new_cache();
        Self { shard, cache }
    }
}

impl Pipeline for LocalPipeline {
    async fn forward(&mut self, pos: usize, hidden: &[f32], n_tokens: usize) -> Result<(Vec<f32>, Vec<HopStat>)> {
        let t = Instant::now();
        let x = self.shard.from_host(hidden, n_tokens)?;
        let y = self.shard.to_host(&self.shard.forward(&mut self.cache, &x, pos)?)?;
        Ok((y, vec![HopStat { busy_us: t.elapsed().as_micros() as u32 }]))
    }

    async fn close(&mut self) -> Result<()> {
        self.cache = self.shard.new_cache();
        Ok(())
    }
}

/// Layers spread over remote nodes arranged in a ring that ends back at this client.
pub struct RemotePipeline {
    net: Arc<Net>,
    rx: UnboundedReceiver<Inbound>,
    first_hop: SocketAddr,
    session: u64,
    codec: CodecKind,
    d_model: usize,
    step: u64,
    timeout: Duration,
}

impl RemotePipeline {
    /// Binds `listen` (port 0 picks a free one) and opens a session through `nodes`. The last
    /// node sends results back to `advertise` (default: the listen IP, or loopback if unspecified).
    pub async fn connect(
        listen: SocketAddr,
        advertise: Option<IpAddr>,
        nodes: &[SocketAddr],
        codec: CodecKind,
        d_model: usize,
        net_opts: NetOpts,
    ) -> Result<Self> {
        let (net, rx) = Net::bind(listen, net_opts)?;
        let ip = advertise.unwrap_or(match listen.ip() {
            ip if ip.is_unspecified() => IpAddr::V4(Ipv4Addr::LOCALHOST),
            ip => ip,
        });
        let me = SocketAddr::new(ip, net.local_addr()?.port());
        Self::open(net, rx, nodes, me, codec, d_model).await
    }

    /// Opens a session along `nodes`; `me` is the address the last node can reach us at.
    pub async fn open(
        net: Arc<Net>,
        mut rx: UnboundedReceiver<Inbound>,
        nodes: &[SocketAddr],
        me: SocketAddr,
        codec: CodecKind,
        d_model: usize,
    ) -> Result<Self> {
        let first_hop = *nodes.first().context("route needs at least one node")?;
        let session = rand::random();
        let mut route = nodes.to_vec();
        route.push(me);
        net.send(first_hop, &Msg::Open { session, route, hop: 0 })?;
        let wait = Duration::from_secs(30);
        let ack = timeout(wait, async {
            loop {
                match rx.recv().await {
                    Some((_, Msg::Open { session: s, .. })) if s == session => return Ok(()),
                    Some((_, Msg::Error { message, hop, .. })) => bail!("hop {hop}: {message}"),
                    Some(_) => continue,
                    None => bail!("network closed"),
                }
            }
        });
        ack.await.context("timed out opening session (is every node up and is --advertise reachable?)")??;
        Ok(Self { net, rx, first_hop, session, codec, d_model, step: 0, timeout: Duration::from_secs(120) })
    }
}

impl Pipeline for RemotePipeline {
    async fn forward(&mut self, pos: usize, hidden: &[f32], n_tokens: usize) -> Result<(Vec<f32>, Vec<HopStat>)> {
        self.step += 1;
        let msg = Msg::Forward(Forward {
            session: self.session,
            step: self.step,
            pos: pos as u32,
            n_tokens: n_tokens as u32,
            codec: self.codec as u8,
            payload: afo_codec::encode(self.codec, hidden, self.d_model),
            hops: Vec::new(),
        });
        self.net.send(self.first_hop, &msg)?;
        let reply = timeout(self.timeout, async {
            loop {
                match self.rx.recv().await {
                    Some((_, Msg::Forward(f))) if f.session == self.session && f.step == self.step => return Ok(f),
                    Some((_, Msg::Error { session, hop, message })) if session == self.session => {
                        bail!("hop {hop}: {message}")
                    }
                    Some(_) => continue,
                    None => bail!("network closed"),
                }
            }
        });
        let f = reply.await.context("timed out waiting for the ring")??;
        let out = afo_codec::decode(self.codec, &f.payload, n_tokens, self.d_model)?;
        Ok((out, f.hops))
    }

    async fn close(&mut self) -> Result<()> {
        self.net.send(self.first_hop, &Msg::Close { session: self.session })
    }
}

#[derive(Debug, Clone)]
pub struct GenOpts {
    pub max_tokens: usize,
    /// 0 means greedy.
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub seed: u64,
    pub stop: Vec<u32>,
    /// Keep the final-position logits of every step (for comparing against the reference).
    pub record_logits: bool,
}

#[derive(Debug, Clone)]
pub struct StepStat {
    pub n_tokens: usize,
    pub wall: Duration,
    /// Embedding, head and sampling on this machine.
    pub local: Duration,
    pub hops: Vec<HopStat>,
}

impl StepStat {
    pub fn busy(&self) -> Duration {
        Duration::from_micros(self.hops.iter().map(|h| h.busy_us as u64).sum())
    }

    /// Wall time not spent computing anywhere: transport, queueing and serialization.
    pub fn network(&self) -> Duration {
        self.wall.saturating_sub(self.local + self.busy())
    }
}

#[derive(Debug, Default)]
pub struct Generation {
    pub tokens: Vec<u32>,
    pub logits: Vec<Vec<f32>>,
    pub steps: Vec<StepStat>,
}

impl Generation {
    pub fn ttft(&self) -> Duration {
        self.steps.first().map_or(Duration::ZERO, |s| s.wall)
    }

    /// Steps after the prefill, i.e. one per generated token after the first.
    pub fn decode_steps(&self) -> &[StepStat] {
        self.steps.get(1..).unwrap_or(&[])
    }

    pub fn decode_tok_per_s(&self) -> f64 {
        let steps = self.decode_steps();
        let t: Duration = steps.iter().map(|s| s.wall).sum();
        if steps.is_empty() { 0. } else { steps.len() as f64 / t.as_secs_f64() }
    }
}

/// Runs the client's ends of the model (embedding, head, sampling) around a pipeline.
pub async fn generate<P: Pipeline>(
    local: &Shard,
    pipeline: &mut P,
    prompt: &[u32],
    opts: &GenOpts,
    mut on_token: impl FnMut(u32),
) -> Result<Generation> {
    let sampling = match (opts.temperature, opts.top_p) {
        (t, _) if t <= 0. => Sampling::ArgMax,
        (temperature, Some(p)) => Sampling::TopP { p, temperature },
        (temperature, None) => Sampling::All { temperature },
    };
    let mut sampler = LogitsProcessor::from_sampling(opts.seed, sampling);
    let mut out = Generation::default();
    let d = local.cfg.d_model;
    let mut input = prompt.to_vec();
    let mut pos = 0;

    while out.tokens.len() < opts.max_tokens {
        let t = Instant::now();
        let n = input.len();
        let hidden = local.to_host(&local.embed(&input)?)?;
        let t_embed = t.elapsed();

        let (hidden, hops) = pipeline.forward(pos, &hidden, n).await?;
        debug_assert_eq!(hidden.len(), n * d);
        let t_back = Instant::now();
        // Only the final position's logits are needed to pick the next token.
        let last = local.from_host(&hidden[(n - 1) * d..], 1)?;
        let logits = local.logits(&last)?.squeeze(0)?;
        let token = sampler.sample(&logits)?;
        if opts.record_logits {
            out.logits.push(logits.to_vec1()?);
        }

        out.steps.push(StepStat { n_tokens: n, wall: t.elapsed(), local: t_embed + t_back.elapsed(), hops });
        out.tokens.push(token);
        on_token(token);
        pos += n;
        input = vec![token];
        if opts.stop.contains(&token) {
            break;
        }
    }
    Ok(out)
}

pub mod chat {
    use anyhow::{Result, anyhow};
    use tokenizers::Tokenizer;

    /// Llama 3 instruct prompt for a single user turn.
    pub fn llama3_prompt(user: &str) -> String {
        format!(
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    }

    /// Llama 3 instruct prompt for a whole conversation of `(role, text)` turns, where role is
    /// `user` or `assistant`, ending ready for the assistant's next reply.
    pub fn llama3_conversation(turns: &[(&str, &str)]) -> String {
        let mut out = String::from("<|begin_of_text|>");
        for (role, text) in turns {
            out += &format!("<|start_header_id|>{role}<|end_header_id|>\n\n{text}<|eot_id|>");
        }
        out + "<|start_header_id|>assistant<|end_header_id|>\n\n"
    }

    pub fn encode(tok: &Tokenizer, text: &str) -> Result<Vec<u32>> {
        Ok(tok.encode(text, false).map_err(|e| anyhow!(e))?.get_ids().to_vec())
    }

    pub fn stop_tokens(tok: &Tokenizer) -> Vec<u32> {
        ["<|eot_id|>", "<|end_of_text|>", "</s>"].iter().filter_map(|t| tok.token_to_id(t)).collect()
    }

    /// Decodes tokens as they stream in, emitting only text that is complete (a multi-byte
    /// character split across tokens is held back until its last byte arrives).
    pub struct TextStream<'a> {
        tok: &'a Tokenizer,
        tokens: Vec<u32>,
        printed: usize,
    }

    impl<'a> TextStream<'a> {
        pub fn new(tok: &'a Tokenizer) -> Self {
            Self { tok, tokens: Vec::new(), printed: 0 }
        }

        pub fn push(&mut self, token: u32) -> Option<String> {
            self.tokens.push(token);
            let text = self.tok.decode(&self.tokens, true).ok()?;
            if text.ends_with('\u{fffd}') {
                return None;
            }
            let new = text.get(self.printed..).filter(|s| !s.is_empty())?.to_string();
            self.printed = text.len();
            Some(new)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::chat;

    #[test]
    fn conversation_matches_single_turn_prompt() {
        assert_eq!(chat::llama3_conversation(&[("user", "hi")]), chat::llama3_prompt("hi"));
        let two = chat::llama3_conversation(&[("user", "a"), ("assistant", "b"), ("user", "c")]);
        assert!(two.starts_with("<|begin_of_text|><|start_header_id|>user"));
        assert!(two.contains("<|start_header_id|>assistant<|end_header_id|>\n\nb<|eot_id|>"));
        assert!(two.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }
}
