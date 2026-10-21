//! Reports what this machine can contribute to a group: run it on every machine and send the
//! resulting JSON file to whoever runs `afo-plan`.

use std::{net::IpAddr, path::PathBuf};

use afo_plan::{
    gb,
    probe::{ProbeOpts, probe},
};
use anyhow::Result;
use clap::Parser;

#[derive(Parser)]
#[command(about = "Measure this machine's memory, disk and speed for planning a group")]
struct Args {
    /// Name shown in the plan (defaults to the hostname).
    #[arg(long)]
    name: Option<String>,
    /// Address the other machines reach this one at (defaults to this machine's Tailscale IP, if any).
    #[arg(long)]
    addr: Option<IpAddr>,
    /// Where model files will be stored; its disk's free space is reported.
    #[arg(long, default_value = "models")]
    models_dir: PathBuf,
    /// Offer at most this much system RAM, in GB.
    #[arg(long)]
    ram_limit_gb: Option<f64>,
    /// Offer at most this much memory on each GPU, in GB.
    #[arg(long)]
    vram_limit_gb: Option<f64>,
    /// Output file (defaults to `<name>.json`).
    #[arg(long)]
    out: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    // Decoding slows down when threads outnumber physical cores, so benchmark the setting the
    // plan's commands use.
    if std::env::var_os("RAYON_NUM_THREADS").is_none() {
        let cores = sysinfo::System::physical_core_count().unwrap_or(1);
        // SAFETY: no other threads exist yet.
        unsafe { std::env::set_var("RAYON_NUM_THREADS", cores.to_string()) };
    }
    let bytes = |g: f64| (g * 1e9) as u64;
    eprintln!("probing (benchmarks each device for a few seconds)...");
    let report = probe(&ProbeOpts {
        name: args.name,
        addr: args.addr,
        models_dir: args.models_dir,
        ram_limit: args.ram_limit_gb.map(bytes),
        vram_limit: args.vram_limit_gb.map(bytes),
    })?;

    eprintln!("{} ({}, {} cores, {} RAM)", report.name, report.cpu, report.cores, gb(report.ram_total));
    match report.addr {
        Some(a) => eprintln!("  address {a}"),
        None => eprintln!("  no address found; pass --addr <ip the others can reach>"),
    }
    eprintln!("  {} free disk for models", gb(report.disk_free));
    for s in &report.slots {
        eprintln!("  {:<7} {:<40} {:>9} usable  {:6.1} GB/s", s.device, s.label, gb(s.mem), s.gbps);
    }
    for w in &report.warnings {
        eprintln!("  warning: {w}");
    }

    let out = args.out.unwrap_or_else(|| PathBuf::from(format!("{}.json", report.name)));
    std::fs::write(&out, serde_json::to_string_pretty(&report)? + "\n")?;
    eprintln!("wrote {}; send it to whoever runs afo-plan", out.display());
    Ok(())
}
