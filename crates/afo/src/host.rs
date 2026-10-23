//! `afo host`: finds the agents on the tailnet, plans the model across them, has each agent load
//! its share, opens the dashboard, and runs a chat in the terminal.

use std::{
    collections::HashMap,
    io::{BufRead, Write},
    net::{IpAddr, Ipv4Addr, SocketAddr},
    process::{Command, Stdio},
    time::Duration,
};

use afo_client::{GenOpts, Pipeline, RemotePipeline, chat, generate};
use afo_codec::CodecKind;
use afo_engine::{LoadOpts, Shard};
use afo_plan::{
    catalog::{self, Candidate},
    gb,
    plan::{AGENT_PORT, Plan, PlanOpts, choose},
    probe::Report,
};
use afo_proto::{NetOpts, http};
use anyhow::{Context, Result, anyhow, bail};
use tokenizers::Tokenizer;
use tokio::net::TcpListener;

use crate::{
    agent::{AgentState, Assignment, NodeSpec, Phase},
    tailscale,
};

const PROBE_TIMEOUT: Duration = Duration::from_secs(3);

pub struct HostOpts {
    /// Use this catalog model instead of choosing one.
    pub model: Option<String>,
    pub min_tok_s: f64,
    pub hop_ms: f64,
    pub ctx: usize,
    pub max_tokens: usize,
    pub temperature: f64,
    /// Agents to use besides those found on the tailnet.
    pub agents: Vec<SocketAddr>,
    /// Don't look for Tailscale; use only this machine and `agents`.
    pub no_tailscale: bool,
    pub dashboard: SocketAddr,
    /// Start without waiting for Enter or asking to confirm the plan.
    pub yes: bool,
}

#[derive(Clone)]
struct Member {
    agent: SocketAddr,
    /// Address the other machines reach this one at.
    ip: IpAddr,
    report: Report,
    is_me: bool,
}

pub async fn run(o: HostOpts) -> Result<()> {
    let me = if o.no_tailscale {
        IpAddr::V4(Ipv4Addr::LOCALHOST)
    } else {
        tailscale::status()?.me
    };

    println!("Looking for machines running the AllForOne agent...");
    let mut members = discover(&o, me).await?;
    print_members(&members);
    if !o.yes {
        println!("\nWaiting for friends to join. Press Enter to plan with these machines.");
        members = lobby(&o, me, members).await?;
    }

    let reports: Vec<Report> = members.iter().map(|m| m.report.clone()).collect();
    let client = members.iter().position(|m| m.is_me).expect("discover always includes this machine");
    let opts = PlanOpts { client, ctx: o.ctx, prompt_tokens: 1024, hop_ms: o.hop_ms };
    let candidates = match &o.model {
        Some(id) => vec![catalog::find(id).with_context(|| format!("{id:?} is not in the catalog"))?],
        None => catalog::candidates(),
    };
    let (results, chosen) = choose(candidates, &reports, &opts, o.min_tok_s, o.model.is_some());
    let Some(chosen) = chosen else {
        println!("\nNothing fits at {} tok/s or more:", o.min_tok_s);
        for (c, r) in results.iter().take(6) {
            let why = r.as_ref().err().cloned().unwrap_or_else(|| "too slow".into());
            println!("  {:<24} {:<7} {why}", c.model.name, c.quant);
        }
        bail!("add machines, lower --min-tok-s, or pick a smaller --model");
    };
    let (c, plan) = (&results[chosen].0, results[chosen].1.as_ref().expect("chosen plan fits"));
    print_plan(c, plan, &members);
    if !o.yes && !confirm("Start? [Y/n] ").await? {
        return Ok(());
    }

    // From here on, make sure every agent is released however the host exits.
    let all: Vec<SocketAddr> = members.iter().map(|m| m.agent).collect();
    let on_ctrl_c = all.clone();
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            println!("\nStopping...");
            release(&on_ctrl_c).await;
            std::process::exit(130);
        }
    });

    let result = serve(&o, c, plan, &members, me).await;
    release(&all).await;
    result
}

async fn serve(o: &HostOpts, c: &Candidate, plan: &Plan, members: &[Member], me: IpAddr) -> Result<()> {
    assign(c, plan, members, o.ctx).await?;
    let states = wait_ready(c, plan, members).await?;

    // Route in ring order, using the ports each agent picked.
    let mut route = Vec::new();
    for n in &plan.nodes {
        let m = members.iter().find(|m| m.report.name == n.machine).expect("plan names a member");
        let port = states[&m.agent]
            .nodes
            .iter()
            .find(|r| r.layers == n.layers)
            .with_context(|| format!("{} isn't running layers {:?}", m.report.name, n.layers))?
            .port;
        route.push(SocketAddr::new(m.ip, port));
    }

    let dashboard = bind_dashboard(o.dashboard).await?;
    let url = format!("http://{}", dashboard.local_addr()?);
    tokio::spawn(afo_monitor::serve(dashboard, route.clone(), Duration::from_secs(1)));
    println!("\nDashboard: {url}");
    open_browser(&url);

    let mine = members.iter().find(|m| m.is_me).expect("this machine is a member");
    let state = &states[&mine.agent];
    let model_path = state.model_path.as_ref().context("this machine's agent has no model file")?;
    let tok_path = state.tokenizer_path.as_ref().context("this machine's agent has no tokenizer")?;
    let tok = Tokenizer::from_file(tok_path).map_err(|e| anyhow!(e))?;
    let device = afo_engine::device(&plan.client_device)?;
    let local = Shard::load(model_path, &LoadOpts { layers: 0..0, embed: true, head: true, max_seq: o.ctx }, &device)?;
    // Any free port: the session tells the last node where to send results, and a fixed port
    // would clash with another `afo host` on this machine.
    let listen = SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 0);
    let mut pipe =
        RemotePipeline::connect(listen, Some(me), &route, CodecKind::F16, local.cfg.d_model, NetOpts::default()).await?;

    println!("\nChatting with {} {}. Type a message; /reset starts over, /exit quits.", c.model.name, c.quant);
    let stop = chat::stop_tokens(&tok);
    let mut turns: Vec<(&'static str, String)> = Vec::new();
    loop {
        let Some(line) = read_line("\nyou> ").await? else { break };
        match line.trim() {
            "" => continue,
            "/exit" | "/quit" => break,
            "/reset" => {
                turns.clear();
                println!("(conversation cleared)");
                continue;
            }
            text => turns.push(("user", text.to_string())),
        }
        let max_tokens = o.max_tokens;
        // Drop the oldest exchanges until the prompt and the reply fit in the context.
        let prompt = loop {
            let refs: Vec<(&str, &str)> = turns.iter().map(|(r, t)| (*r, t.as_str())).collect();
            let ids = chat::encode(&tok, &chat::llama3_conversation(&refs))?;
            if ids.len() + max_tokens <= o.ctx || turns.len() <= 1 {
                break ids;
            }
            turns.drain(..2.min(turns.len() - 1));
        };
        let opts = GenOpts {
            max_tokens: max_tokens.min(o.ctx.saturating_sub(prompt.len())),
            temperature: o.temperature,
            top_p: Some(0.9),
            seed: rand_seed(),
            stop: stop.clone(),
            record_logits: false,
        };
        print!("\nassistant> ");
        std::io::stdout().flush().ok();
        let mut stream = chat::TextStream::new(&tok);
        let generation = generate(&local, &mut pipe, &prompt, &opts, |t| {
            if let Some(s) = stream.push(t) {
                print!("{s}");
                std::io::stdout().flush().ok();
            }
        })
        .await?;
        let reply_ids: Vec<u32> = generation.tokens.iter().copied().filter(|t| !stop.contains(t)).collect();
        let reply = tok.decode(&reply_ids, true).map_err(|e| anyhow!(e))?;
        println!("\n  ({} tokens, {:.1} tok/s)", generation.tokens.len(), generation.decode_tok_per_s());
        turns.push(("assistant", reply));
    }
    pipe.close().await.ok();
    Ok(())
}

async fn discover(o: &HostOpts, me: IpAddr) -> Result<Vec<Member>> {
    let mine = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), AGENT_PORT);
    let mut targets = vec![(mine, me, true)];
    if !o.no_tailscale {
        for ip in tailscale::status()?.peers {
            targets.push((SocketAddr::new(ip, AGENT_PORT), ip, false));
        }
    }
    targets.extend(o.agents.iter().map(|&a| (a, a.ip(), false)));

    let probes: Vec<_> = targets
        .into_iter()
        .map(|(agent, ip, is_me)| {
            tokio::spawn(async move {
                let body = http::get(agent, "/probe", PROBE_TIMEOUT).await?;
                let report: Report = serde_json::from_slice(&body)?;
                Ok::<_, anyhow::Error>(Member { agent, ip, report, is_me })
            })
        })
        .collect();
    let mut members: Vec<Member> = Vec::new();
    for (i, p) in probes.into_iter().enumerate() {
        match p.await? {
            Ok(mut m) => {
                m.report.addr = Some(m.ip);
                // The planner tells machines apart by name.
                let base = m.report.name.clone();
                let mut n = 2;
                while members.iter().any(|o| o.report.name == m.report.name) {
                    m.report.name = format!("{base}-{n}");
                    n += 1;
                }
                members.push(m);
            }
            Err(e) if i == 0 => {
                return Err(e.context("no agent is running on this machine; start it with `afo agent`"));
            }
            Err(_) => {} // a tailnet machine without AllForOne
        }
    }
    Ok(members)
}

/// Re-scans every few seconds, printing changes, until Enter is pressed.
async fn lobby(o: &HostOpts, me: IpAddr, mut members: Vec<Member>) -> Result<Vec<Member>> {
    let mut enter = tokio::task::spawn_blocking(|| std::io::stdin().lock().read_line(&mut String::new()));
    loop {
        tokio::select! {
            _ = &mut enter => return Ok(members),
            _ = tokio::time::sleep(Duration::from_secs(3)) => {
                let found = discover(o, me).await?;
                let names = |ms: &[Member]| ms.iter().map(|m| m.report.name.clone()).collect::<Vec<_>>();
                if names(&found) != names(&members) {
                    print_members(&found);
                    members = found;
                }
            }
        }
    }
}

fn print_members(members: &[Member]) {
    println!("\n{} machine(s):", members.len());
    for m in members {
        let slots: Vec<String> =
            m.report.slots.iter().map(|s| format!("{} {} @ {:.0} GB/s", s.device, gb(s.mem), s.gbps)).collect();
        let you = if m.is_me { " (this machine)" } else { "" };
        println!("  {}{you}  {}  {}", m.report.name, m.ip, slots.join(", "));
        for w in &m.report.warnings {
            println!("    note: {w}");
        }
    }
}

fn print_plan(c: &Candidate, plan: &Plan, members: &[Member]) {
    println!(
        "\nBest fit: {} {} ({}), about {:.1} tokens/s",
        c.model.name,
        c.quant,
        gb(c.file_bytes),
        plan.tok_per_s()
    );
    for m in members {
        let mine: Vec<String> = plan
            .nodes
            .iter()
            .filter(|n| n.machine == m.report.name)
            .map(|n| format!("layers {}-{} on {}", n.layers.start, n.layers.end - 1, n.device))
            .collect();
        let role = if m.is_me { " + chat" } else { "" };
        let work = if mine.is_empty() { "not needed".to_string() } else { mine.join(", ") };
        println!("  {}: {work}{role}", m.report.name);
    }
}

async fn assign(c: &Candidate, plan: &Plan, members: &[Member], ctx: usize) -> Result<()> {
    for m in members {
        let nodes: Vec<NodeSpec> = plan
            .nodes
            .iter()
            .filter(|n| n.machine == m.report.name)
            .map(|n| NodeSpec { layers: n.layers.clone(), device: n.device.clone() })
            .collect();
        // This machine downloads the model even without layers: the chat needs its ends.
        if nodes.is_empty() && !m.is_me {
            http::post(m.agent, "/release", b"", PROBE_TIMEOUT).await.ok();
            continue;
        }
        let a = Assignment { model: c.id(), max_seq: ctx, nodes, tokenizer: m.is_me };
        http::post(m.agent, "/assign", &serde_json::to_vec(&a)?, PROBE_TIMEOUT)
            .await
            .with_context(|| format!("assigning work to {}", m.report.name))?;
    }
    Ok(())
}

/// Polls the working agents, printing progress, until all are ready or one fails.
async fn wait_ready(c: &Candidate, plan: &Plan, members: &[Member]) -> Result<HashMap<SocketAddr, AgentState>> {
    let working: Vec<&Member> =
        members.iter().filter(|m| m.is_me || plan.nodes.iter().any(|n| n.machine == m.report.name)).collect();
    println!("\nPreparing {} on {} machine(s):", c.id(), working.len());
    let mut last: HashMap<SocketAddr, String> = HashMap::new();
    loop {
        let mut states = HashMap::new();
        for m in &working {
            let body = http::get(m.agent, "/state", PROBE_TIMEOUT).await.with_context(|| format!("asking {}", m.report.name))?;
            let s: AgentState = serde_json::from_slice(&body)?;
            // `key` changes when the phase does, or every 10% of a download, so progress is
            // printed without flooding the screen.
            let (key, line) = match &s.phase {
                Phase::Idle => ("waiting".to_string(), "waiting".to_string()),
                Phase::Downloading { done, total } => {
                    let pct = *done as f64 / *total as f64 * 100.;
                    (format!("dl{}", pct as u64 / 10), format!("downloading {} of {} ({pct:.0}%)", gb(*done), gb(*total)))
                }
                Phase::Starting => ("starting".into(), "loading layers".into()),
                Phase::Ready => ("ready".into(), "ready".into()),
                Phase::Failed { error } => bail!("{} failed: {error}", m.report.name),
            };
            if last.get(&m.agent) != Some(&key) {
                println!("  {}: {line}", m.report.name);
                last.insert(m.agent, key);
            }
            states.insert(m.agent, s);
        }
        if states.values().all(|s| s.phase == Phase::Ready) {
            return Ok(states);
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}

async fn release(agents: &[SocketAddr]) {
    for &a in agents {
        http::post(a, "/release", b"", PROBE_TIMEOUT).await.ok();
    }
}

async fn bind_dashboard(addr: SocketAddr) -> Result<TcpListener> {
    // Fall back to a nearby port if a standalone afo-monitor already has this one.
    for port in addr.port()..addr.port() + 10 {
        if let Ok(l) = TcpListener::bind(SocketAddr::new(addr.ip(), port)).await {
            return Ok(l);
        }
    }
    bail!("no free port for the dashboard near {addr}")
}

fn open_browser(url: &str) {
    let mut cmd = if cfg!(target_os = "windows") {
        let mut c = Command::new("cmd");
        c.args(["/C", "start", "", url]);
        c
    } else if cfg!(target_os = "macos") {
        let mut c = Command::new("open");
        c.arg(url);
        c
    } else {
        let mut c = Command::new("xdg-open");
        c.arg(url);
        c
    };
    // Not fatal: the URL is printed for opening by hand.
    let _ = cmd.stdout(Stdio::null()).stderr(Stdio::null()).spawn();
}

async fn read_line(prompt: &str) -> Result<Option<String>> {
    print!("{prompt}");
    std::io::stdout().flush().ok();
    tokio::task::spawn_blocking(|| {
        let mut s = String::new();
        Ok(match std::io::stdin().lock().read_line(&mut s)? {
            0 => None,
            _ => Some(s),
        })
    })
    .await?
}

async fn confirm(prompt: &str) -> Result<bool> {
    Ok(read_line(prompt).await?.is_some_and(|s| !s.trim().to_lowercase().starts_with('n')))
}

fn rand_seed() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64)
}
