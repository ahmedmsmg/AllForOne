//! Standalone dashboard for a ring started by hand; `afo host` serves the same page itself.

use std::{
    collections::HashSet,
    net::{IpAddr, SocketAddr},
    path::PathBuf,
    time::Duration,
};

use anyhow::{Context, Result, ensure};
use clap::Parser;
use serde::Deserialize;
use tokio::net::TcpListener;

#[derive(Parser)]
#[command(about = "Serve a live web dashboard of the ring's nodes")]
struct Args {
    /// Node addresses, as in the client's --route.
    #[arg(long, value_delimiter = ',')]
    nodes: Vec<SocketAddr>,
    /// Take node addresses from `afo-plan --json` output instead.
    #[arg(long)]
    plan: Option<PathBuf>,
    /// Where to serve the page. Use 0.0.0.0:7300 to open it from other devices.
    #[arg(long, default_value = "127.0.0.1:7300")]
    listen: SocketAddr,
    #[arg(long, default_value_t = 1000)]
    interval_ms: u64,
}

#[derive(Deserialize)]
struct PlanFile {
    nodes: Vec<PlanNode>,
}

#[derive(Deserialize)]
struct PlanNode {
    machine: String,
    addr: Option<IpAddr>,
    port: u16,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter(
        tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
    ).init();
    let args = Args::parse();

    let mut addrs = args.nodes.clone();
    if let Some(path) = &args.plan {
        let plan: PlanFile = serde_json::from_str(&std::fs::read_to_string(path)?)
            .with_context(|| format!("parsing {} (expected afo-plan --json output)", path.display()))?;
        for n in plan.nodes {
            let ip = n.addr.with_context(|| format!("plan has no address for {}; pass --nodes", n.machine))?;
            addrs.push(SocketAddr::new(ip, n.port));
        }
    }
    let mut seen = HashSet::new();
    addrs.retain(|a| seen.insert(*a));
    ensure!(!addrs.is_empty(), "no nodes to watch; pass --nodes or --plan");

    let listener = TcpListener::bind(args.listen).await.with_context(|| format!("binding {}", args.listen))?;
    println!("dashboard on http://{}", listener.local_addr()?);
    afo_monitor::serve(listener, addrs, Duration::from_millis(args.interval_ms)).await;
    Ok(())
}
