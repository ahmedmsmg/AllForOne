//! Picks the best model a group's pooled hardware can run and prints each person's commands.

use std::path::PathBuf;

use afo_plan::{
    catalog::{self, Candidate},
    gb,
    plan::{Plan, PlanOpts, evaluate},
    probe::{REPORT_VERSION, Report},
};
use anyhow::{Context, Result, bail, ensure};
use clap::Parser;

#[derive(Parser)]
#[command(about = "Choose a model and layer split from the machines' afo-probe reports")]
struct Args {
    /// Report files written by afo-probe, one per machine.
    #[arg(required = true)]
    reports: Vec<PathBuf>,
    /// Machine that runs the client (defaults to the first report).
    #[arg(long)]
    client: Option<String>,
    /// Context length (prompt plus output) to reserve KV cache for.
    #[arg(long, default_value_t = 4096)]
    ctx: usize,
    /// Longest prompt to reserve prefill memory for.
    #[arg(long, default_value_t = 1024)]
    prompt_tokens: usize,
    /// Assumed one-way network latency between machines, in ms.
    #[arg(long, default_value_t = 30.)]
    hop_ms: f64,
    /// Skip models estimated to decode slower than this.
    #[arg(long, default_value_t = 2.)]
    min_tok_s: f64,
    /// Plan this catalog entry (e.g. `llama-3.1-8b:Q4_K_M`) instead of choosing one.
    #[arg(long)]
    model: Option<String>,
    /// Directory the commands download models into.
    #[arg(long, default_value = "models")]
    models_dir: String,
    /// Print the chosen plan as JSON instead of instructions.
    #[arg(long)]
    json: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut reports = Vec::new();
    for path in &args.reports {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let r: Report = serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        ensure!(r.version == REPORT_VERSION, "{} is report version {}, expected {REPORT_VERSION}", path.display(), r.version);
        ensure!(!reports.iter().any(|o: &Report| o.name == r.name), "two reports are named {:?}", r.name);
        reports.push(r);
    }
    let client = match &args.client {
        Some(name) => reports.iter().position(|r| &r.name == name).with_context(|| format!("no report named {name:?}"))?,
        None => 0,
    };
    let opts = PlanOpts { client, ctx: args.ctx, prompt_tokens: args.prompt_tokens, hop_ms: args.hop_ms };

    let candidates = match &args.model {
        Some(id) => vec![catalog::find(id).with_context(|| format!("{id:?} is not in the catalog"))?],
        None => catalog::candidates(),
    };
    let results: Vec<(Candidate, Result<Plan, String>)> =
        candidates.into_iter().map(|c| (c, evaluate(&c, &reports, &opts))).collect();
    let chosen = results.iter().find_map(|(c, r)| match r {
        Ok(p) if args.model.is_some() || p.tok_per_s() >= args.min_tok_s => Some((c, p)),
        _ => None,
    });

    if args.json {
        let Some((_, plan)) = chosen else { bail!("no model fits") };
        println!("{}", serde_json::to_string_pretty(plan)?);
        return Ok(());
    }

    print_pool(&reports, client);
    println!("\nCandidates, best first ({} ms per hop assumed, at least {} tok/s wanted):", args.hop_ms, args.min_tok_s);
    for (c, r) in &results {
        let mark = if chosen.is_some_and(|(k, _)| k.id() == c.id()) { "->" } else { "  " };
        let verdict = match r {
            Ok(p) => format!("fits, ~{:.1} tok/s over {} node(s)", p.tok_per_s(), p.nodes.len()),
            Err(why) => format!("no: {why}"),
        };
        println!("{mark} {:<24} {:<7} {:>8}  {verdict}", c.model.name, c.quant, gb(c.file_bytes));
    }
    let Some((c, plan)) = chosen else {
        bail!("nothing fits at {} tok/s or more; lower --min-tok-s or --ctx, or add machines", args.min_tok_s);
    };
    print_plan(c, plan, &reports, &args);
    Ok(())
}

fn print_pool(reports: &[Report], client: usize) {
    println!("Pool:");
    for (i, r) in reports.iter().enumerate() {
        let role = if i == client { " (client)" } else { "" };
        let addr = r.addr.map_or("<no address>".into(), |a| a.to_string());
        println!("  {}{role}  {addr}  {} free disk", r.name, gb(r.disk_free));
        for s in &r.slots {
            println!("    {:<7} {:<40} {:>9}  {:6.1} GB/s", s.device, s.label, gb(s.mem), s.gbps);
        }
        for w in &r.warnings {
            println!("    warning: {w}");
        }
    }
}

fn print_plan(c: &Candidate, plan: &Plan, reports: &[Report], args: &Args) {
    let dir = &args.models_dir;
    let (model, tokenizer) = (format!("{dir}/{}", c.file_name()), format!("{dir}/{}", c.tokenizer_name()));
    let addr = |name: &str| {
        reports.iter().find(|r| r.name == name).and_then(|r| r.addr).map_or(format!("<{name}-ip>"), |a| a.to_string())
    };

    println!(
        "\nPlan: {} {}, {} layers over {} node(s), ~{:.1} tok/s (layers {:.0} ms + network {:.0} ms + client {:.0} ms per token)",
        c.model.name,
        c.quant,
        c.model.n_layer,
        plan.nodes.len(),
        plan.tok_per_s(),
        plan.compute_ms,
        plan.network_ms,
        plan.client_ms,
    );
    let report = |name: &str| reports.iter().find(|r| r.name == name).expect("plan names a reported machine");
    let get_model = |sh: Shell| {
        format!("  {}{}{}", sh.mkdir(dir), sh.and(), sh.download(&c.url(), &model))
    };
    println!("\nEach machine downloads the model ({}) into {dir}/, then starts its node(s).", gb(c.file_bytes));

    let mut machines: Vec<&str> = plan.nodes.iter().map(|n| n.machine.as_str()).collect();
    machines.dedup();
    for &m in &machines {
        let r = report(m);
        let sh = Shell::of(r);
        println!("\n{m} ({}):", sh.name());
        println!("{}", get_model(sh));
        for n in plan.nodes.iter().filter(|n| n.machine == m) {
            // CPU decoding is fastest with one thread per physical core, which is what afo-probe measured.
            let threads = if n.device == "cpu" { sh.env("RAYON_NUM_THREADS", r.cores) } else { String::new() };
            println!(
                "  {threads}afo-node --model {model} --layers {}..{} --listen 0.0.0.0:{} --device {} --max-seq {}",
                n.layers.start, n.layers.end, n.port, n.device, args.ctx
            );
        }
    }

    let client = report(&plan.client);
    let sh = Shell::of(client);
    let route: Vec<String> = plan.nodes.iter().map(|n| format!("{}:{}", addr(&n.machine), n.port)).collect();
    println!("\n{} (client, {}), once every node prints \"listening\":", plan.client, sh.name());
    if !machines.contains(&plan.client.as_str()) {
        println!("{}", get_model(sh));
    }
    println!("  {}", sh.download(&c.tokenizer_url(), &tokenizer));
    let threads = if plan.client_device == "cpu" { sh.env("RAYON_NUM_THREADS", client.cores) } else { String::new() };
    let nl = sh.continuation();
    println!(
        "  {threads}afo-client --model {model} --tokenizer {tokenizer} --device {} --max-seq {}{nl}--route {}{nl}--advertise {} --listen 0.0.0.0:{} --prompt \"Hello\"",
        plan.client_device,
        args.ctx,
        route.join(","),
        addr(&plan.client),
        afo_plan::plan::BASE_PORT + 100,
    );
}

/// Command syntax for the shell each machine's commands are printed for.
#[derive(Clone, Copy)]
enum Shell {
    Posix,
    PowerShell,
}

impl Shell {
    fn of(r: &Report) -> Self {
        if r.os.starts_with("Windows") { Shell::PowerShell } else { Shell::Posix }
    }

    fn name(self) -> &'static str {
        match self {
            Shell::Posix => "shell",
            Shell::PowerShell => "PowerShell",
        }
    }

    fn env(self, var: &str, value: impl std::fmt::Display) -> String {
        match self {
            Shell::Posix => format!("{var}={value} "),
            Shell::PowerShell => format!("$env:{var}={value}; "),
        }
    }

    fn mkdir(self, dir: &str) -> String {
        match self {
            Shell::Posix => format!("mkdir -p {dir}"),
            Shell::PowerShell => format!("New-Item -ItemType Directory -Force {dir} | Out-Null"),
        }
    }

    fn and(self) -> &'static str {
        match self {
            Shell::Posix => " && ",
            Shell::PowerShell => "; ",
        }
    }

    /// Windows PowerShell aliases `curl` to a different command, so name the real one.
    fn download(self, url: &str, path: &str) -> String {
        match self {
            Shell::Posix => format!("curl -L -o {path} {url}"),
            Shell::PowerShell => format!("curl.exe -L -o {path} {url}"),
        }
    }

    fn continuation(self) -> &'static str {
        match self {
            Shell::Posix => " \\\n    ",
            Shell::PowerShell => " `\n    ",
        }
    }
}
