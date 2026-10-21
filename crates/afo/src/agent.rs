//! The agent runs in the background on every machine. It reports the machine's hardware to the
//! host, and on the host's instruction downloads the model and runs node processes for its share
//! of the layers. Only loopback and Tailscale addresses may talk to it.

use std::{
    fs::File,
    net::SocketAddr,
    ops::Range,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use afo_plan::{
    catalog::{self, Candidate},
    plan::{AGENT_PORT, BASE_PORT},
    probe::{ProbeOpts, Report, probe},
};
use afo_proto::{
    http::{self, Request, Response},
    status::STATUS_PATH,
};
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use tracing::{info, warn};

use crate::tailscale;

/// How long a node may take to load its layers before the agent gives up on it.
const LOAD_TIMEOUT: Duration = Duration::from_secs(900);

/// What the host asks an agent to run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Assignment {
    /// Catalog id, e.g. `llama-3.1-8b:Q4_K_M`. Agents only download catalog models.
    pub model: String,
    pub max_seq: usize,
    pub nodes: Vec<NodeSpec>,
    /// Also fetch the tokenizer (the machine running the chat needs it).
    pub tokenizer: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeSpec {
    pub layers: Range<usize>,
    pub device: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "phase", rename_all = "snake_case")]
pub enum Phase {
    Idle,
    Downloading { done: u64, total: u64 },
    Starting,
    Ready,
    Failed { error: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunningNode {
    pub layers: Range<usize>,
    pub device: String,
    pub port: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    pub phase: Phase,
    pub model: Option<String>,
    pub model_path: Option<PathBuf>,
    pub tokenizer_path: Option<PathBuf>,
    pub nodes: Vec<RunningNode>,
}

pub struct AgentOpts {
    pub port: u16,
    pub node_port: u16,
    pub data_dir: PathBuf,
    pub name: Option<String>,
    pub ram_limit: Option<u64>,
    pub vram_limit: Option<u64>,
}

impl Default for AgentOpts {
    fn default() -> Self {
        Self { port: AGENT_PORT, node_port: BASE_PORT, data_dir: data_dir(), name: None, ram_limit: None, vram_limit: None }
    }
}

/// `~/.allforone`, where models and logs live.
pub fn data_dir() -> PathBuf {
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).unwrap_or_else(|| ".".into());
    PathBuf::from(home).join(".allforone")
}

struct Shared {
    report: Report,
    state: AgentState,
    children: Vec<Child>,
    /// Bumped by every assign or release, so a superseded run stops touching the state.
    generation: u64,
}

struct Agent {
    opts: AgentOpts,
    shared: Mutex<Shared>,
}

pub async fn run(opts: AgentOpts) -> Result<()> {
    std::fs::create_dir_all(opts.data_dir.join("models"))?;
    std::fs::create_dir_all(opts.data_dir.join("logs"))?;
    info!("measuring this machine (a few seconds)...");
    let models_dir = opts.data_dir.join("models");
    let (name, ram_limit, vram_limit) = (opts.name.clone(), opts.ram_limit, opts.vram_limit);
    let report = tokio::task::spawn_blocking(move || {
        probe(&ProbeOpts { name, addr: None, models_dir, ram_limit, vram_limit })
    })
    .await??;
    for s in &report.slots {
        info!("{}: {} usable, {:.1} GB/s", s.device, afo_plan::gb(s.mem), s.gbps);
    }
    for w in &report.warnings {
        warn!("{w}");
    }

    let listener = TcpListener::bind(SocketAddr::from(([0, 0, 0, 0], opts.port)))
        .await
        .with_context(|| format!("binding port {} (is another agent already running?)", opts.port))?;
    info!("agent for {} ready on port {}", report.name, opts.port);
    let idle = AgentState { phase: Phase::Idle, model: None, model_path: None, tokenizer_path: None, nodes: vec![] };
    let agent = Arc::new(Agent { opts, shared: Mutex::new(Shared { report, state: idle, children: vec![], generation: 0 }) });
    let handler = agent.clone();
    http::serve(listener, move |req| handler.handle(req)).await;
    Ok(())
}

impl Agent {
    fn handle(self: &Arc<Self>, req: &Request) -> Response {
        let ip = req.peer.ip();
        if !(ip.is_loopback() || tailscale::is_tailnet(ip)) {
            return Response::error(403, "only this machine and Tailscale peers may use the agent");
        }
        match (req.method.as_str(), req.path.as_str()) {
            ("GET", "/probe") => json(&self.shared.lock().unwrap().report),
            ("GET", "/state") => json(&self.shared.lock().unwrap().state),
            ("POST", "/assign") => match serde_json::from_slice::<Assignment>(&req.body) {
                Ok(a) => match catalog::find(&a.model) {
                    Some(c) => {
                        self.assign(c, a);
                        Response::json("\"ok\"".into())
                    }
                    None => Response::error(400, &format!("{:?} is not a catalog model", a.model)),
                },
                Err(e) => Response::error(400, &format!("bad assignment: {e}")),
            },
            ("POST", "/release") => {
                self.release();
                Response::json("\"ok\"".into())
            }
            _ => Response::not_found(),
        }
    }

    /// Stops whatever runs now and starts `a` in the background.
    fn assign(self: &Arc<Self>, c: Candidate, a: Assignment) {
        let generation = {
            let mut s = self.shared.lock().unwrap();
            stop_children(&mut s);
            s.generation += 1;
            // Stays "starting" when the model is already on disk; the download reports progress.
            s.state = AgentState {
                phase: Phase::Starting,
                model: Some(c.id()),
                model_path: None,
                tokenizer_path: None,
                nodes: vec![],
            };
            s.generation
        };
        info!("assigned {} layers {:?}", c.id(), a.nodes.iter().map(|n| &n.layers).collect::<Vec<_>>());
        let agent = self.clone();
        tokio::spawn(async move {
            if let Err(e) = agent.start(c, a, generation).await {
                warn!("assignment failed: {e:#}");
                agent.update(generation, |s| s.phase = Phase::Failed { error: format!("{e:#}") });
            }
        });
    }

    fn release(&self) {
        let mut s = self.shared.lock().unwrap();
        stop_children(&mut s);
        s.generation += 1;
        s.state = AgentState { phase: Phase::Idle, model: None, model_path: None, tokenizer_path: None, nodes: vec![] };
        info!("released");
    }

    /// Applies `f` to the state unless a newer assignment or release has replaced this run.
    fn update(&self, generation: u64, f: impl FnOnce(&mut AgentState)) -> bool {
        let mut s = self.shared.lock().unwrap();
        if s.generation != generation {
            return false;
        }
        f(&mut s.state);
        true
    }

    async fn start(self: &Arc<Self>, c: Candidate, a: Assignment, generation: u64) -> Result<()> {
        let models = self.opts.data_dir.join("models");
        let model_path = models.join(c.file_name());
        download(&c.url(), &model_path, Some(c.file_bytes), |done| {
            self.update(generation, |s| s.phase = Phase::Downloading { done, total: c.file_bytes })
        })
        .await?;
        let tokenizer_path = if a.tokenizer {
            let path = models.join(c.tokenizer_name());
            download(&c.tokenizer_url(), &path, None, |_| self.update(generation, |_| ())).await?;
            Some(path)
        } else {
            None
        };
        if !self.update(generation, |s| {
            s.phase = Phase::Starting;
            s.model_path = Some(model_path.clone());
            s.tokenizer_path = tokenizer_path.clone();
        }) {
            return Ok(());
        }

        let cores = self.shared.lock().unwrap().report.cores;
        let node_exe = sibling_exe("afo-node")?;
        let mut running = Vec::new();
        for (i, spec) in a.nodes.iter().enumerate() {
            let port = self.opts.node_port + i as u16;
            let log_path = self.opts.data_dir.join("logs").join(format!("node-{port}.log"));
            let log = File::create(&log_path)?;
            let mut cmd = Command::new(&node_exe);
            cmd.arg("--model")
                .arg(&model_path)
                .args(["--layers", &format!("{}..{}", spec.layers.start, spec.layers.end)])
                .args(["--listen", &format!("0.0.0.0:{port}")])
                .args(["--device", &spec.device])
                .args(["--max-seq", &a.max_seq.to_string()])
                .stdin(Stdio::null())
                .stdout(log.try_clone()?)
                .stderr(log);
            // CPU decoding is fastest with one thread per physical core.
            if spec.device == "cpu" {
                cmd.env("RAYON_NUM_THREADS", cores.to_string());
            }
            let child = cmd.spawn().with_context(|| format!("starting {}", node_exe.display()))?;
            let mut s = self.shared.lock().unwrap();
            if s.generation != generation {
                let mut child = child;
                let _ = child.kill();
                return Ok(());
            }
            s.children.push(child);
            running.push((RunningNode { layers: spec.layers.clone(), device: spec.device.clone(), port }, log_path));
        }

        // A node is ready once it answers its status port.
        let t = Instant::now();
        for (i, (node, log_path)) in running.iter().enumerate() {
            let status = SocketAddr::from(([127, 0, 0, 1], node.port));
            loop {
                if http::get(status, STATUS_PATH, Duration::from_secs(2)).await.is_ok() {
                    break;
                }
                let exited = {
                    let mut s = self.shared.lock().unwrap();
                    if s.generation != generation {
                        return Ok(());
                    }
                    s.children.get_mut(i).and_then(|c| c.try_wait().ok().flatten())
                };
                if let Some(code) = exited {
                    bail!("node for layers {:?} exited ({code}):\n{}", node.layers, log_tail(log_path));
                }
                ensure!(t.elapsed() < LOAD_TIMEOUT, "node for layers {:?} did not start in time", node.layers);
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
        }
        let nodes = running.into_iter().map(|(n, _)| n).collect();
        self.update(generation, |s| {
            s.phase = Phase::Ready;
            s.nodes = nodes;
        });
        info!("ready");
        Ok(())
    }
}

fn stop_children(s: &mut Shared) {
    for mut c in s.children.drain(..) {
        let _ = c.kill();
        let _ = c.wait();
    }
}

/// Fetches `url` to `path` with the system `curl`, resuming a partial `.part` file, and reports
/// progress through `progress`; stops early (without error) once `progress` returns false.
async fn download(url: &str, path: &Path, size: Option<u64>, progress: impl Fn(u64) -> bool) -> Result<()> {
    let len = |p: &Path| std::fs::metadata(p).map_or(0, |m| m.len());
    if path.exists() && size.is_none_or(|s| len(path) == s) {
        return Ok(());
    }
    let part = path.with_extension("part");
    if size.is_none_or(|s| len(&part) != s) {
        info!("downloading {url}");
        let mut child = tokio::process::Command::new("curl")
            .args(["-fsSL", "--retry", "5", "-C", "-", "-o"])
            .arg(&part)
            .arg(url)
            .kill_on_drop(true)
            .spawn()
            .context("running curl (is it installed?)")?;
        let status = loop {
            tokio::select! {
                status = child.wait() => break status?,
                _ = tokio::time::sleep(Duration::from_millis(500)) => {
                    if !progress(len(&part)) {
                        return Ok(());
                    }
                }
            }
        };
        ensure!(status.success(), "downloading {url} failed ({status})");
    }
    if let Some(s) = size {
        ensure!(len(&part) == s, "download of {url} is {} bytes, expected {s}", len(&part));
    }
    std::fs::rename(&part, path)?;
    progress(len(path));
    Ok(())
}

/// Another of the installed programs, next to this one.
fn sibling_exe(name: &str) -> Result<PathBuf> {
    let me = std::env::current_exe()?;
    let path = me.with_file_name(format!("{name}{}", std::env::consts::EXE_SUFFIX));
    ensure!(path.exists(), "{} not found next to {}", path.display(), me.display());
    Ok(path)
}

fn log_tail(path: &Path) -> String {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    let lines: Vec<&str> = text.lines().collect();
    lines[lines.len().saturating_sub(8)..].join("\n")
}

fn json(v: &impl Serialize) -> Response {
    Response::json(serde_json::to_string(v).expect("agent types serialize"))
}
