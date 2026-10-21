//! Checks split execution against a single-machine reference and measures where time goes.
//!
//! Runs greedy decoding once with every layer local (the reference), then once per codec through
//! the node ring, and reports token agreement, logit drift and latency for each.

use std::{
    net::{IpAddr, SocketAddr},
    path::PathBuf,
    time::Duration,
};

use afo_client::{GenOpts, Generation, LocalPipeline, Pipeline, RemotePipeline, chat, generate};
use afo_codec::CodecKind;
use afo_engine::{LoadOpts, Shard};
use anyhow::{Result, anyhow};
use clap::Parser;
use tokenizers::Tokenizer;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    tokenizer: PathBuf,
    /// Node addresses in layer order. Without it, only the reference runs.
    #[arg(long, value_delimiter = ',')]
    route: Vec<SocketAddr>,
    #[arg(long, value_delimiter = ',', default_value = "f32,f16,fp8,int8")]
    codecs: Vec<CodecKind>,
    #[arg(long, default_value = "Explain in three sentences why the sky is blue.")]
    prompt: String,
    #[arg(long, default_value_t = 48)]
    max_tokens: usize,
    #[arg(long, default_value = "cpu")]
    device: String,
    #[arg(long, default_value = "0.0.0.0:0")]
    listen: SocketAddr,
    #[arg(long)]
    advertise: Option<IpAddr>,
    #[arg(long, default_value_t = 0)]
    sim_delay_ms: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("warn").init();
    let args = Args::parse();
    let tok = Tokenizer::from_file(&args.tokenizer).map_err(|e| anyhow!(e))?;
    let device = afo_engine::device(&args.device)?;
    let max_seq = 4096;
    let local = Shard::load(&args.model, &LoadOpts { layers: 0..0, embed: true, head: true, max_seq }, &device)?;
    let n_layer = local.cfg.n_layer;

    let prompt = chat::encode(&tok, &chat::llama3_prompt(&args.prompt))?;
    // Ignore stop tokens so every run produces the same number of steps.
    let opts = GenOpts {
        max_tokens: args.max_tokens,
        temperature: 0.,
        top_p: None,
        seed: 0,
        stop: vec![],
        record_logits: true,
    };

    let reference = {
        let full = Shard::load(&args.model, &LoadOpts { layers: 0..n_layer, embed: false, head: false, max_seq }, &device)?;
        let mut pipe = LocalPipeline::new(full);
        generate(&local, &mut pipe, &prompt, &opts, |_| {}).await?
    };
    println!("prompt: {} tokens; reference output:\n{}\n", prompt.len(), tok.decode(&reference.tokens, true).map_err(|e| anyhow!(e))?);
    println!("{:<10} {:>8} {:>12} {:>12} {:>9} {:>11} {:>11}", "run", "match", "max |dlogit|", "1st diverge", "tok/s", "TTFT ms", "net ms/tok");
    row("reference", &reference, &reference);

    for &codec in &args.codecs {
        if args.route.is_empty() {
            break;
        }
        let net_opts = afo_proto::NetOpts { sim_delay: Duration::from_millis(args.sim_delay_ms) };
        let mut pipe =
            RemotePipeline::connect(args.listen, args.advertise, &args.route, codec, local.cfg.d_model, net_opts).await?;
        let g = generate(&local, &mut pipe, &prompt, &opts, |_| {}).await?;
        pipe.close().await?;
        row(&format!("ring {codec}"), &g, &reference);
    }
    Ok(())
}

fn row(name: &str, g: &Generation, reference: &Generation) {
    let diverge = g.tokens.iter().zip(&reference.tokens).position(|(a, b)| a != b);
    // Logits are comparable only while both runs have seen the same tokens.
    let comparable = diverge.map_or(g.logits.len(), |i| i + 1);
    let max_diff = g.logits[..comparable]
        .iter()
        .zip(&reference.logits)
        .flat_map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x - y).abs()))
        .fold(0f32, f32::max);
    let steps = g.decode_steps();
    let net_ms = steps.iter().map(|s| s.network().as_secs_f64() * 1e3).sum::<f64>() / steps.len().max(1) as f64;
    let matched = diverge.unwrap_or(g.tokens.len());
    println!(
        "{name:<10} {:>8} {max_diff:>12.4} {:>12} {:>9.2} {:>11.1} {net_ms:>11.2}",
        format!("{matched}/{}", g.tokens.len()),
        diverge.map_or("-".into(), |i| i.to_string()),
        g.decode_tok_per_s(),
        g.ttft().as_secs_f64() * 1e3,
    );
}
