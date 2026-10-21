//! Layer-sliced quantized Llama inference.
//!
//! A [`Shard`] loads only the transformer blocks in its layer range (plus, optionally, the token
//! embedding and the output head) from a GGUF file, so a node never holds weights it doesn't run.
//! The KV cache lives outside the shard in a [`KvCache`], one per session.

use std::{collections::HashMap, fs::File, ops::Range, path::Path};

use anyhow::{Context, Result, bail, ensure};
use candle_core::{
    DType, Device, Module, Tensor,
    quantized::{QMatMul, gguf_file},
};
use candle_nn::Embedding;
use candle_transformers::{
    quantized_nn::RmsNorm,
    utils::{build_causal_mask, repeat_kv},
};

#[derive(Debug, Clone)]
pub struct Config {
    pub n_layer: usize,
    pub d_model: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub rope_base: f32,
    pub eps: f64,
    pub vocab: usize,
    pub max_seq: usize,
}

impl Config {
    fn from_gguf(ct: &gguf_file::Content, max_seq: usize) -> Result<Self> {
        let arch = ct
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .cloned()
            .unwrap_or_default();
        // Only the plain Llama graph (Llama 2/3, Mistral, ...) is implemented so far.
        ensure!(arch == "llama", "unsupported architecture {arch:?} (only \"llama\")");
        let md = |k: &str| {
            ct.metadata
                .get(&format!("llama.{k}"))
                .with_context(|| format!("missing llama.{k} in GGUF metadata"))
        };
        let n_head = md("attention.head_count")?.to_u32()? as usize;
        let d_model = md("embedding_length")?.to_u32()? as usize;
        let ctx_len = md("context_length").and_then(|v| Ok(v.to_u32()? as usize)).unwrap_or(4096);
        let vocab = ct
            .tensor_infos
            .get("token_embd.weight")
            .context("missing token_embd.weight")?
            .shape
            .dims()[0];
        Ok(Self {
            n_layer: md("block_count")?.to_u32()? as usize,
            d_model,
            n_head,
            n_kv_head: md("attention.head_count_kv")?.to_u32()? as usize,
            head_dim: d_model / n_head,
            rope_dim: md("rope.dimension_count")?.to_u32()? as usize,
            rope_base: md("rope.freq_base").and_then(|v| Ok(v.to_f32()?)).unwrap_or(10000.),
            eps: md("attention.layer_norm_rms_epsilon")?.to_f32()? as f64,
            vocab,
            max_seq: max_seq.min(ctx_len),
        })
    }
}

/// Reads model hyperparameters without loading any tensors.
pub fn read_config(path: &Path, max_seq: usize) -> Result<Config> {
    let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let ct = gguf_file::Content::read(&mut file)?;
    Config::from_gguf(&ct, max_seq)
}

pub fn device(spec: &str) -> Result<Device> {
    Ok(match spec {
        "cpu" => Device::Cpu,
        "metal" => Device::new_metal(0)?,
        s if s.starts_with("cuda") => {
            let ord = s.strip_prefix("cuda:").map(str::parse).transpose()?.unwrap_or(0);
            Device::new_cuda(ord)?
        }
        s => bail!("unknown device {s:?} (cpu | cuda[:N] | metal)"),
    })
}

struct Block {
    attn_norm: RmsNorm,
    wq: QMatMul,
    wk: QMatMul,
    wv: QMatMul,
    wo: QMatMul,
    ffn_norm: RmsNorm,
    w_gate: QMatMul,
    w_up: QMatMul,
    w_down: QMatMul,
}

/// Per-session attention state for the blocks of one shard.
pub struct KvCache {
    kv: Vec<Option<(Tensor, Tensor)>>,
    len: usize,
}

impl KvCache {
    /// Number of positions currently cached.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Drops every cached position at or after `pos` (used to roll back rejected draft tokens).
    fn truncate(&mut self, pos: usize) -> Result<()> {
        if pos >= self.len {
            return Ok(());
        }
        for slot in &mut self.kv {
            *slot = match slot.take() {
                Some((k, v)) if pos > 0 => Some((k.narrow(2, 0, pos)?, v.narrow(2, 0, pos)?)),
                _ => None,
            };
        }
        self.len = pos;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct LoadOpts {
    pub layers: Range<usize>,
    pub embed: bool,
    pub head: bool,
    pub max_seq: usize,
}

pub struct Shard {
    pub cfg: Config,
    pub layers: Range<usize>,
    device: Device,
    blocks: Vec<Block>,
    embed: Option<Embedding>,
    head: Option<(RmsNorm, QMatMul)>,
    cos: Tensor,
    sin: Tensor,
}

impl Shard {
    pub fn load(path: &Path, opts: &LoadOpts, device: &Device) -> Result<Self> {
        let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let ct = gguf_file::Content::read(&mut file)?;
        let cfg = Config::from_gguf(&ct, opts.max_seq)?;
        let range = opts.layers.clone();
        ensure!(
            range.start <= range.end && range.end <= cfg.n_layer,
            "layer range {range:?} outside model's 0..{}",
            cfg.n_layer
        );

        let mut get = |name: &str| ct.tensor(&mut file, name, device);
        let qmm = |t| QMatMul::from_qtensor(t);
        let norm = |t, eps| RmsNorm::from_qtensor(t, eps);

        let mut blocks = Vec::with_capacity(range.len());
        for i in range.clone() {
            let p = format!("blk.{i}");
            blocks.push(Block {
                attn_norm: norm(get(&format!("{p}.attn_norm.weight"))?, cfg.eps)?,
                wq: qmm(get(&format!("{p}.attn_q.weight"))?)?,
                wk: qmm(get(&format!("{p}.attn_k.weight"))?)?,
                wv: qmm(get(&format!("{p}.attn_v.weight"))?)?,
                wo: qmm(get(&format!("{p}.attn_output.weight"))?)?,
                ffn_norm: norm(get(&format!("{p}.ffn_norm.weight"))?, cfg.eps)?,
                w_gate: qmm(get(&format!("{p}.ffn_gate.weight"))?)?,
                w_up: qmm(get(&format!("{p}.ffn_up.weight"))?)?,
                w_down: qmm(get(&format!("{p}.ffn_down.weight"))?)?,
            });
        }

        let embed_q = if opts.embed || opts.head { Some(get("token_embd.weight")?) } else { None };
        let embed = match (&embed_q, opts.embed) {
            (Some(q), true) => Some(Embedding::new(q.dequantize(device)?, cfg.d_model)),
            _ => None,
        };
        let head = if opts.head {
            // Tied embeddings (e.g. Llama 3.2 1B/3B) have no separate output matrix.
            let out = match get("output.weight") {
                Ok(t) => t,
                Err(_) => embed_q.context("no output.weight and no token_embd.weight")?,
            };
            Some((norm(get("output_norm.weight")?, cfg.eps)?, qmm(out)?))
        } else {
            None
        };

        // Llama 3 stores its RoPE frequency scaling as per-dimension divisors.
        let freq_factors = match get("rope_freqs.weight") {
            Ok(t) => Some(t.dequantize(device)?.to_dtype(DType::F32)?.to_vec1::<f32>()?),
            Err(_) => None,
        };
        let (cos, sin) = rope_tables(&cfg, freq_factors.as_deref(), device)?;

        Ok(Self { cfg, layers: range, device: device.clone(), blocks, embed, head, cos, sin })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn new_cache(&self) -> KvCache {
        KvCache { kv: (0..self.blocks.len()).map(|_| None).collect(), len: 0 }
    }

    /// Token ids -> hidden states of shape `(1, n, d_model)`.
    pub fn embed(&self, tokens: &[u32]) -> Result<Tensor> {
        let embed = self.embed.as_ref().context("shard was loaded without the embedding")?;
        let ids = Tensor::new(tokens, &self.device)?.unsqueeze(0)?;
        Ok(embed.forward(&ids)?)
    }

    /// Runs this shard's blocks over `x` (shape `(1, n, d_model)`) placed at positions
    /// `pos..pos+n`. A `pos` before the end of the cache discards the cached positions after it.
    pub fn forward(&self, cache: &mut KvCache, x: &Tensor, pos: usize) -> Result<Tensor> {
        ensure!(pos <= cache.len, "position {pos} skips ahead of KV cache length {}", cache.len);
        let (_, seq_len, _) = x.dims3()?;
        ensure!(pos + seq_len <= self.cfg.max_seq, "context exceeds max_seq {}", self.cfg.max_seq);
        cache.truncate(pos)?;
        let mask = if seq_len > 1 { Some(build_causal_mask(seq_len, pos, &self.device)?) } else { None };
        let cos = self.cos.narrow(0, pos, seq_len)?;
        let sin = self.sin.narrow(0, pos, seq_len)?;

        let mut x = x.clone();
        for (block, slot) in self.blocks.iter().zip(cache.kv.iter_mut()) {
            let h = block.attn_norm.forward(&x)?;
            let h = self.attention(block, slot, &h, mask.as_ref(), &cos, &sin)?;
            x = (x + h)?;
            let h = block.ffn_norm.forward(&x)?;
            let gate = candle_nn::ops::silu(&block.w_gate.forward(&h)?)?;
            let h = block.w_down.forward(&(gate * block.w_up.forward(&h)?)?)?;
            x = (x + h)?;
        }
        cache.len = pos + seq_len;
        Ok(x)
    }

    fn attention(
        &self,
        b: &Block,
        slot: &mut Option<(Tensor, Tensor)>,
        x: &Tensor,
        mask: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let c = &self.cfg;
        let (bsz, seq_len, _) = x.dims3()?;
        let heads = |t: Tensor, n| t.reshape((bsz, seq_len, n, c.head_dim))?.transpose(1, 2)?.contiguous();
        let q = heads(b.wq.forward(x)?, c.n_head)?;
        let k = heads(b.wk.forward(x)?, c.n_kv_head)?;
        let v = heads(b.wv.forward(x)?, c.n_kv_head)?;
        let q = candle_nn::rotary_emb::rope_i(&q, cos, sin)?;
        let k = candle_nn::rotary_emb::rope_i(&k, cos, sin)?;

        let (k, v) = match slot.take() {
            Some((kc, vc)) => (Tensor::cat(&[&kc, &k], 2)?, Tensor::cat(&[&vc, &v], 2)?),
            None => (k, v),
        };
        *slot = Some((k.clone(), v.clone()));

        let n_rep = c.n_head / c.n_kv_head;
        let k = repeat_kv(k, n_rep)?;
        let v = repeat_kv(v, n_rep)?.contiguous()?;
        let att = (q.matmul(&k.t()?)? / (c.head_dim as f64).sqrt())?;
        let att = match mask {
            Some(m) => {
                let m = m.broadcast_as(att.shape())?;
                let neg_inf = Tensor::new(f32::NEG_INFINITY, &self.device)?.broadcast_as(att.shape())?;
                m.where_cond(&neg_inf, &att)?
            }
            None => att,
        };
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let y = att.matmul(&v)?.transpose(1, 2)?.reshape((bsz, seq_len, c.n_head * c.head_dim))?;
        Ok(b.wo.forward(&y)?)
    }

    /// Hidden states `(1, n, d_model)` -> logits `(n, vocab)` for every position.
    pub fn logits(&self, x: &Tensor) -> Result<Tensor> {
        let (norm, out) = self.head.as_ref().context("shard was loaded without the output head")?;
        Ok(out.forward(&norm.forward(x)?)?.squeeze(0)?.to_dtype(DType::F32)?)
    }

    pub fn to_host(&self, x: &Tensor) -> Result<Vec<f32>> {
        Ok(x.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?)
    }

    pub fn from_host(&self, data: &[f32], n_tokens: usize) -> Result<Tensor> {
        ensure!(data.len() == n_tokens * self.cfg.d_model, "hidden buffer has wrong length");
        Ok(Tensor::from_slice(data, (1, n_tokens, self.cfg.d_model), &self.device)?)
    }
}

fn rope_tables(cfg: &Config, freq_factors: Option<&[f32]>, device: &Device) -> Result<(Tensor, Tensor)> {
    let theta: Vec<f32> = (0..cfg.rope_dim)
        .step_by(2)
        .enumerate()
        .map(|(j, i)| {
            let f = 1. / cfg.rope_base.powf(i as f32 / cfg.rope_dim as f32);
            freq_factors.map_or(f, |ff| f / ff[j])
        })
        .collect();
    let n = theta.len();
    let theta = Tensor::from_vec(theta, (1, n), device)?;
    let pos = Tensor::arange(0, cfg.max_seq as u32, device)?.to_dtype(DType::F32)?.reshape((cfg.max_seq, 1))?;
    let idx = pos.matmul(&theta)?;
    Ok((idx.cos()?, idx.sin()?))
}

/// What a pipeline node runs: hidden states in, hidden states out, with per-session KV state.
/// Keeps the network layer independent of the inference backend.
pub trait Engine: Send {
    fn hidden_size(&self) -> usize;
    fn forward(&mut self, session: u64, pos: usize, hidden: &[f32], n_tokens: usize) -> Result<Vec<f32>>;
    fn close(&mut self, session: u64);
}

pub struct CandleEngine {
    shard: Shard,
    sessions: HashMap<u64, KvCache>,
}

impl CandleEngine {
    pub fn new(shard: Shard) -> Self {
        Self { shard, sessions: HashMap::new() }
    }
}

impl Engine for CandleEngine {
    fn hidden_size(&self) -> usize {
        self.shard.cfg.d_model
    }

    fn forward(&mut self, session: u64, pos: usize, hidden: &[f32], n_tokens: usize) -> Result<Vec<f32>> {
        let cache = self.sessions.entry(session).or_insert_with(|| self.shard.new_cache());
        let x = self.shard.from_host(hidden, n_tokens)?;
        let y = self.shard.forward(cache, &x, pos)?;
        self.shard.to_host(&y)
    }

    fn close(&mut self, session: u64) {
        self.sessions.remove(&session);
    }
}

/// Parses a half-open layer range written `start..end`.
pub fn parse_range(s: &str) -> Result<Range<usize>> {
    let (a, b) = s.split_once("..").context("expected a range like 0..8")?;
    let (a, b): (usize, usize) = (a.trim().parse()?, b.trim().parse()?);
    ensure!(a < b, "empty layer range {s}");
    Ok(a..b)
}
