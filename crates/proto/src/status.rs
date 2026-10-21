//! What a node reports at `GET /status` on the TCP port matching its QUIC (UDP) port.

use std::ops::Range;

use serde::{Deserialize, Serialize};

pub const STATUS_PATH: &str = "/status";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeStatus {
    pub host: String,
    /// Model file name.
    pub model: String,
    pub n_layer: usize,
    pub layers: Range<usize>,
    pub device: String,
    pub uptime_s: f64,
    pub sessions: usize,
    /// Forward passes run, and tokens they covered (prompt and generated).
    pub steps: u64,
    pub tokens: u64,
    pub errors: u64,
    /// Generated tokens per second over the last few seconds, summed over sessions. Every
    /// generated token passes every node, so this is also the ring's generation rate.
    pub decode_tok_s: f64,
    /// Share of recent wall time spent computing.
    pub busy_pct: f64,
    pub last_step_ms: Option<f64>,
    /// Seconds since the last forward pass.
    pub idle_s: Option<f64>,
    pub res: Resources,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Resources {
    /// This process's CPU use as a share of the whole machine.
    pub cpu_pct: f32,
    /// This process's resident memory.
    pub rss: u64,
    pub ram_used: u64,
    pub ram_total: u64,
    pub gpu: Option<GpuStatus>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuStatus {
    pub name: String,
    pub util_pct: f32,
    pub mem_used: u64,
    pub mem_total: u64,
}
