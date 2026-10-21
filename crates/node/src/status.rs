//! Activity counters and resource sampling behind the node's `GET /status`.

use std::{
    collections::VecDeque,
    ops::Range,
    process::Command,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use afo_proto::status::{GpuStatus, NodeStatus, Resources};
use sysinfo::{ProcessRefreshKind, ProcessesToUpdate, System};

/// Rates are averaged over this much recent history.
const WINDOW: Duration = Duration::from_secs(10);
/// With no step for this long, the generation rate reads as zero.
const IDLE_AFTER: Duration = Duration::from_secs(5);
const SAMPLE_EVERY: Duration = Duration::from_secs(1);

/// What this node runs; fixed for the life of the process.
pub struct Info {
    pub host: String,
    pub model: String,
    pub n_layer: usize,
    pub layers: Range<usize>,
    pub device: String,
}

struct Step {
    at: Instant,
    n_tokens: u32,
    busy: Duration,
}

pub struct Activity {
    started: Instant,
    pub sessions: usize,
    steps: u64,
    tokens: u64,
    errors: u64,
    last: Option<Step>,
    recent: VecDeque<Step>,
}

impl Activity {
    pub fn new() -> Self {
        Self { started: Instant::now(), sessions: 0, steps: 0, tokens: 0, errors: 0, last: None, recent: VecDeque::new() }
    }

    pub fn record(&mut self, n_tokens: u32, busy: Duration) {
        let at = Instant::now();
        self.steps += 1;
        self.tokens += n_tokens as u64;
        self.last = Some(Step { at, n_tokens, busy });
        self.recent.push_back(Step { at, n_tokens, busy });
        self.trim(at);
    }

    pub fn error(&mut self) {
        self.errors += 1;
    }

    fn trim(&mut self, now: Instant) {
        while self.recent.front().is_some_and(|s| now.duration_since(s.at) > WINDOW) {
            self.recent.pop_front();
        }
    }

    pub fn snapshot(&mut self, info: &Info, res: &Resources) -> NodeStatus {
        let now = Instant::now();
        self.trim(now);
        let idle = self.last.as_ref().map(|s| now.duration_since(s.at));
        // Single-token steps are generated tokens; longer ones are prompt prefill.
        let decode: Vec<Instant> = self.recent.iter().filter(|s| s.n_tokens == 1).map(|s| s.at).collect();
        let decode_tok_s = match (decode.first(), decode.last()) {
            (Some(&a), Some(&b)) if decode.len() >= 2 && idle.is_some_and(|i| i < IDLE_AFTER) => {
                (decode.len() - 1) as f64 / b.duration_since(a).as_secs_f64()
            }
            _ => 0.,
        };
        let span = WINDOW.min(now.duration_since(self.started)).as_secs_f64();
        let busy = self.recent.iter().fold(0., |t, s| t + s.busy.as_secs_f64());
        NodeStatus {
            host: info.host.clone(),
            model: info.model.clone(),
            n_layer: info.n_layer,
            layers: info.layers.clone(),
            device: info.device.clone(),
            uptime_s: now.duration_since(self.started).as_secs_f64(),
            sessions: self.sessions,
            steps: self.steps,
            tokens: self.tokens,
            errors: self.errors,
            decode_tok_s,
            busy_pct: if span > 0. { (busy / span * 100.).min(100.) } else { 0. },
            last_step_ms: self.last.as_ref().map(|s| s.busy.as_secs_f64() * 1e3),
            idle_s: idle.map(|i| i.as_secs_f64()),
            res: res.clone(),
        }
    }
}

/// Refreshes `out` every second on a background thread, so answering a status request never
/// waits on `nvidia-smi`.
pub fn spawn_sampler(device: &str, out: Arc<Mutex<Resources>>) {
    let gpu = device.strip_prefix("cuda").map(|i| i.trim_start_matches(':').parse::<usize>().unwrap_or(0));
    std::thread::spawn(move || {
        let pid = sysinfo::get_current_pid().ok();
        let cpus = std::thread::available_parallelism().map_or(1, |n| n.get()) as f32;
        let mut sys = System::new();
        loop {
            sys.refresh_memory();
            // Process CPU share is measured against system CPU time, which needs its own refresh.
            sys.refresh_cpu_usage();
            let mut res = Resources { ram_used: sys.used_memory(), ram_total: sys.total_memory(), ..Default::default() };
            if let Some(pid) = pid {
                let what = ProcessRefreshKind::nothing().with_cpu().with_memory();
                sys.refresh_processes_specifics(ProcessesToUpdate::Some(&[pid]), true, what);
                if let Some(p) = sys.process(pid) {
                    res.cpu_pct = p.cpu_usage() / cpus;
                    res.rss = p.memory();
                }
            }
            res.gpu = gpu.and_then(nvidia_status);
            *out.lock().unwrap() = res;
            std::thread::sleep(SAMPLE_EVERY);
        }
    });
}

fn nvidia_status(index: usize) -> Option<GpuStatus> {
    let out = Command::new("nvidia-smi")
        .args([
            &format!("--id={index}"),
            "--query-gpu=name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    let text = String::from_utf8(out.stdout).ok()?;
    let fields: Vec<&str> = text.trim().split(',').map(str::trim).collect();
    let [name, util, used, total] = fields[..] else { return None };
    Some(GpuStatus {
        name: name.into(),
        util_pct: util.parse().ok()?,
        mem_used: used.parse::<u64>().ok()? << 20,
        mem_total: total.parse::<u64>().ok()? << 20,
    })
}
