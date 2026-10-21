//! Fits a model onto the pooled devices and splits its layers among them.
//!
//! Decoding a token runs every layer once, one after another, so per-token time is the sum of
//! each layer's time on its device plus one network hop per machine boundary in the ring. The
//! planner therefore fills the fastest devices first, and tries leaving machines out when the hop
//! they add costs more than the layers they would take.

use std::{net::IpAddr, ops::Range};

use serde::Serialize;

use crate::{catalog::Candidate, gb, probe::Report};

/// Runtime, allocator slack and (on GPUs) driver context per node process.
const NODE_OVERHEAD: u64 = 512 << 20;
/// Same for the client process, on top of its embedding and head.
const CLIENT_OVERHEAD: u64 = 512 << 20;
/// Port of the agent on every machine (TCP), which `afo host` talks to.
pub const AGENT_PORT: u16 = 7100;
/// First port for node processes; a machine running several gets consecutive ports.
pub const BASE_PORT: u16 = 7101;
/// Port the client listens on for results coming back around the ring.
pub const CLIENT_PORT: u16 = 7200;
/// Beyond this many machines, all of them are used instead of searching subsets.
const MAX_SUBSET_SEARCH: usize = 12;

pub struct PlanOpts {
    /// Index of the report for the machine that runs the client.
    pub client: usize,
    /// Context length (prompt plus generated tokens) to reserve KV cache for.
    pub ctx: usize,
    /// Longest prompt to reserve prefill scratch for.
    pub prompt_tokens: usize,
    /// Assumed one-way latency between machines.
    pub hop_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Node {
    pub machine: String,
    pub addr: Option<IpAddr>,
    pub device: String,
    pub port: u16,
    pub layers: Range<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Plan {
    pub model: String,
    pub client: String,
    pub client_device: String,
    /// In ring order.
    pub nodes: Vec<Node>,
    pub compute_ms: f64,
    pub network_ms: f64,
    pub client_ms: f64,
}

impl Plan {
    pub fn tok_per_s(&self) -> f64 {
        1e3 / (self.compute_ms + self.network_ms + self.client_ms)
    }
}

/// One device's share of the work for a particular model.
struct Capacity {
    machine: usize,
    slot: usize,
    layers: usize,
    ms_per_layer: f64,
}

/// Every candidate's plan (or why it doesn't fit), and the index of the one to use: the best that
/// fits and decodes at least `min_tok_s`. With `forced`, the first that fits, whatever its speed.
pub fn choose(
    candidates: Vec<Candidate>,
    reports: &[Report],
    o: &PlanOpts,
    min_tok_s: f64,
    forced: bool,
) -> (Vec<(Candidate, Result<Plan, String>)>, Option<usize>) {
    let results: Vec<_> = candidates.into_iter().map(|c| (c, evaluate(&c, reports, o))).collect();
    let chosen = results.iter().position(|(_, r)| r.as_ref().is_ok_and(|p| forced || p.tok_per_s() >= min_tok_s));
    (results, chosen)
}

/// Plans `c` onto the machines in `reports`, or explains why it doesn't fit.
pub fn evaluate(c: &Candidate, reports: &[Report], o: &PlanOpts) -> Result<Plan, String> {
    let client = &reports[o.client];
    if client.disk_free < c.file_bytes {
        return Err(format!("{} has {} free disk, needs {}", client.name, gb(client.disk_free), gb(c.file_bytes)));
    }
    let Some(host) = client.slots.iter().position(|s| s.is_host()) else {
        return Err(format!("{} reported no CPU or Metal device to run the client on", client.name));
    };

    let per_layer = c.layer_bytes() + c.kv_bytes_per_layer(o.ctx);
    // The KV cache grows by concatenation, so appending briefly holds one layer's cache twice.
    let reserve = NODE_OVERHEAD + c.prefill_bytes(o.prompt_tokens) + c.kv_bytes_per_layer(o.ctx);
    let client_reserve = c.client_bytes() + CLIENT_OVERHEAD;
    if client.slots[host].mem < client_reserve {
        return Err(format!("{} needs {} for the client, has {}", client.name, gb(client_reserve), gb(client.slots[host].mem)));
    }

    let mut caps = Vec::new();
    let mut offered = 0;
    for (m, r) in reports.iter().enumerate() {
        if r.disk_free < c.file_bytes {
            continue;
        }
        for (s, slot) in r.slots.iter().enumerate() {
            let taken = if m == o.client && s == host { client_reserve } else { 0 };
            let usable = slot.mem.saturating_sub(taken).saturating_sub(reserve);
            offered += usable;
            let layers = (usable / per_layer) as usize;
            if layers > 0 {
                caps.push(Capacity { machine: m, slot: s, layers, ms_per_layer: c.layer_bytes() as f64 / (slot.gbps * 1e6) });
            }
        }
    }
    let n_layer = c.model.n_layer;
    if caps.iter().map(|k| k.layers).sum::<usize>() < n_layer {
        return Err(format!("needs ~{} for layers and cache, pool offers {}", gb(n_layer as u64 * per_layer), gb(offered)));
    }

    let mut machines: Vec<usize> = caps.iter().map(|k| k.machine).filter(|&m| m != o.client).collect();
    machines.dedup();
    let subsets: Vec<u64> = if machines.len() <= MAX_SUBSET_SEARCH {
        (0..1u64 << machines.len()).collect()
    } else {
        vec![(1u64 << machines.len()) - 1]
    };

    let mut best: Option<(f64, Vec<(usize, usize)>)> = None;
    for mask in subsets {
        let chosen = |m: usize| m == o.client || machines.iter().position(|&x| x == m).is_some_and(|i| mask >> i & 1 == 1);
        let mut usable: Vec<(usize, &Capacity)> = caps.iter().enumerate().filter(|(_, k)| chosen(k.machine)).collect();
        usable.sort_by(|a, b| a.1.ms_per_layer.total_cmp(&b.1.ms_per_layer));
        let (mut left, mut compute, mut fill) = (n_layer, 0., Vec::new());
        for (i, k) in usable {
            let n = left.min(k.layers);
            if n == 0 {
                break;
            }
            left -= n;
            compute += n as f64 * k.ms_per_layer;
            fill.push((i, n));
        }
        if left > 0 {
            continue;
        }
        let total = compute + network_ms(&fill, &caps, o);
        if best.as_ref().is_none_or(|(t, _)| total < *t) {
            best = Some((total, fill));
        }
    }
    let (_, mut fill) = best.expect("all machines together have enough capacity");

    // Ring order: the client's own devices first (no network hop), then machine by machine.
    fill.sort_by_key(|&(i, _)| (caps[i].machine != o.client, caps[i].machine, caps[i].slot));
    let mut nodes = Vec::new();
    let (mut start, mut port, mut prev) = (0, BASE_PORT, usize::MAX);
    for &(i, n) in &fill {
        let k = &caps[i];
        port = if k.machine == prev { port + 1 } else { BASE_PORT };
        prev = k.machine;
        let r = &reports[k.machine];
        nodes.push(Node {
            machine: r.name.clone(),
            addr: r.addr,
            device: r.slots[k.slot].device.clone(),
            port,
            layers: start..start + n,
        });
        start += n;
    }

    let host = &client.slots[host];
    Ok(Plan {
        model: c.id(),
        client: client.name.clone(),
        client_device: host.device.clone(),
        compute_ms: fill.iter().map(|&(i, n)| n as f64 * caps[i].ms_per_layer).sum(),
        network_ms: network_ms(&fill, &caps, o),
        client_ms: c.head_bytes() as f64 / (host.gbps * 1e6),
        nodes,
    })
}

/// Hops in the ring client -> nodes -> client that cross between machines. Nodes on the client's
/// own machine are visited first, so each other machine adds one hop, plus one to get back.
fn network_ms(fill: &[(usize, usize)], caps: &[Capacity], o: &PlanOpts) -> f64 {
    let mut remote: Vec<usize> = fill.iter().map(|&(i, _)| caps[i].machine).filter(|&m| m != o.client).collect();
    remote.sort_unstable();
    remote.dedup();
    if remote.is_empty() { 0. } else { (remote.len() + 1) as f64 * o.hop_ms }
}
