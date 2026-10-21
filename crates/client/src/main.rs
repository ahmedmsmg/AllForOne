//! Chat client: embeds the prompt locally, sends hidden states around the node ring, samples
//! from the returned states locally, and prints latency stats.

use std::{
    io::Write,
    net::{IpAddr, SocketAddr},
    path::PathBuf,
    time::Duration,
};

use afo_client::{GenOpts, Generation, RemotePipeline, chat, generate, Pipeline};
use afo_codec::CodecKind;
use afo_engine::{LoadOpts, Shard};
use afo_proto::NetOpts;
use anyhow::{Result, anyhow};
use clap::Parser;
use tokenizers::Tokenizer;

#[derive(Parser)]
#[command(about = "Generate text with a model split across pipeline nodes")]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    tokenizer: PathBuf,
    /// Node addresses in layer order, comma-separated.
    #[arg(long, value_delimiter = ',', required = true)]
    route: Vec<SocketAddr>,
    #[arg(long, default_value = "0.0.0.0:0")]
    listen: SocketAddr,
    /// IP the last node should send results to (defaults to the listen IP, or loopback).
    #[arg(long)]
    advertise: Option<IpAddr>,
    /// Activation encoding on the wire: f32 | f16 | fp8 | int8
    #[arg(long, default_value = "f16")]
    codec: CodecKind,
    #[arg(long, default_value = "cpu")]
    device: String,
    #[arg(long)]
    prompt: String,
    /// Send the prompt as-is instead of wrapping it in the Llama 3 chat template.
    #[arg(long)]
    raw: bool,
    #[arg(long, default_value_t = 128)]
    max_tokens: usize,
    #[arg(long, default_value_t = 0.0)]
    temperature: f64,
    #[arg(long)]
    top_p: Option<f64>,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 4096)]
    max_seq: usize,
    #[arg(long, default_value_t = 0)]
    sim_delay_ms: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter(
        tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()),
    ).init();
    let args = Args::parse();

    let tok = Tokenizer::from_file(&args.tokenizer).map_err(|e| anyhow!(e))?;
    let device = afo_engine::device(&args.device)?;
    let local = Shard::load(&args.model, &LoadOpts { layers: 0..0, embed: true, head: true, max_seq: args.max_seq }, &device)?;

    let net_opts = NetOpts { sim_delay: Duration::from_millis(args.sim_delay_ms) };
    let mut pipe = RemotePipeline::connect(args.listen, args.advertise, &args.route, args.codec, local.cfg.d_model, net_opts).await?;

    let text = if args.raw { args.prompt.clone() } else { chat::llama3_prompt(&args.prompt) };
    let prompt = chat::encode(&tok, &text)?;
    let opts = GenOpts {
        max_tokens: args.max_tokens,
        temperature: args.temperature,
        top_p: args.top_p,
        seed: args.seed,
        stop: chat::stop_tokens(&tok),
        record_logits: false,
    };

    let mut stream = chat::TextStream::new(&tok);
    let generation = generate(&local, &mut pipe, &prompt, &opts, |t| {
        if let Some(s) = stream.push(t) {
            print!("{s}");
            std::io::stdout().flush().ok();
        }
    })
    .await;
    pipe.close().await?;
    println!();
    report(&generation?, prompt.len(), args.route.len());
    Ok(())
}

fn report(g: &Generation, prompt_len: usize, n_nodes: usize) {
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    let steps = g.decode_steps();
    let avg = |f: &dyn Fn(&afo_client::StepStat) -> Duration| {
        if steps.is_empty() { 0. } else { steps.iter().map(|s| ms(f(s))).sum::<f64>() / steps.len() as f64 }
    };
    eprintln!("---");
    eprintln!("prompt {prompt_len} tokens, generated {} tokens, {n_nodes} nodes", g.tokens.len());
    eprintln!("time to first token  {:8.1} ms", ms(g.ttft()));
    eprintln!("decode               {:8.2} tok/s", g.decode_tok_per_s());
    eprintln!(
        "per decode step      {:8.1} ms = nodes {:.1} + client {:.1} + network {:.1}",
        avg(&|s| s.wall),
        avg(&|s| s.busy()),
        avg(&|s| s.local),
        avg(&|s| s.network()),
    );
}
