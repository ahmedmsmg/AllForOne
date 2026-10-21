//! `afo`: everyday use of AllForOne. Every machine runs `afo agent` in the background (the
//! installer sets that up); whoever wants to chat runs `afo host`.

mod agent;
mod host;
mod tailscale;

use std::{net::SocketAddr, path::PathBuf};

use afo_plan::plan::{AGENT_PORT, BASE_PORT};
use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(about = "Run a large model across your friends' computers", version)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Offer this machine to the group (runs in the background; the installer starts it).
    Agent {
        /// Name shown to the host (defaults to the hostname).
        #[arg(long)]
        name: Option<String>,
        /// Offer at most this much system RAM, in GB.
        #[arg(long)]
        ram_limit_gb: Option<f64>,
        /// Offer at most this much memory on each GPU, in GB.
        #[arg(long)]
        vram_limit_gb: Option<f64>,
        /// Where models and logs are kept (default ~/.allforone).
        #[arg(long)]
        data_dir: Option<PathBuf>,
        #[arg(long, default_value_t = AGENT_PORT)]
        port: u16,
        /// First port for node processes.
        #[arg(long, default_value_t = BASE_PORT)]
        node_port: u16,
    },
    /// Gather the machines on your tailnet, load the best model they can run, and chat with it.
    Host {
        /// Use this model instead of choosing (e.g. llama-3.1-8b:Q4_K_M).
        #[arg(long)]
        model: Option<String>,
        /// Skip models estimated to generate slower than this.
        #[arg(long, default_value_t = 2.)]
        min_tok_s: f64,
        /// Assumed one-way network latency between machines, in ms.
        #[arg(long, default_value_t = 30.)]
        hop_ms: f64,
        /// Context length (prompt plus reply) in tokens.
        #[arg(long, default_value_t = 4096)]
        ctx: usize,
        /// Longest reply, in tokens.
        #[arg(long, default_value_t = 1024)]
        max_tokens: usize,
        #[arg(long, default_value_t = 0.7)]
        temperature: f64,
        /// Extra agents to use (host:port), e.g. ones not on Tailscale.
        #[arg(long, value_delimiter = ',')]
        agents: Vec<SocketAddr>,
        /// Don't use Tailscale: only this machine and --agents.
        #[arg(long)]
        no_tailscale: bool,
        #[arg(long, default_value = "127.0.0.1:7300")]
        dashboard: SocketAddr,
        /// Start right away, without waiting for Enter or confirming the plan.
        #[arg(long, short)]
        yes: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let level = if matches!(cli.cmd, Cmd::Agent { .. }) { "info" } else { "warn" };
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| level.into()))
        .init();
    let bytes = |g: f64| (g * 1e9) as u64;
    match cli.cmd {
        Cmd::Agent { name, ram_limit_gb, vram_limit_gb, data_dir, port, node_port } => {
            agent::run(agent::AgentOpts {
                port,
                node_port,
                data_dir: data_dir.unwrap_or_else(agent::data_dir),
                name,
                ram_limit: ram_limit_gb.map(bytes),
                vram_limit: vram_limit_gb.map(bytes),
            })
            .await
        }
        Cmd::Host { model, min_tok_s, hop_ms, ctx, max_tokens, temperature, agents, no_tailscale, dashboard, yes } => {
            host::run(host::HostOpts {
                model,
                min_tok_s,
                hop_ms,
                ctx,
                max_tokens,
                temperature,
                agents,
                no_tailscale,
                dashboard,
                yes,
            })
            .await
        }
    }
}
