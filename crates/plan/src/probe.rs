//! Measures what one machine can offer: usable memory per device, free disk, and how fast each
//! device runs a decode step.

use std::{
    net::IpAddr,
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, Instant},
};

use anyhow::{Result, ensure};
use candle_core::{
    Device, Module, Tensor,
    quantized::{GgmlDType, QMatMul, QTensor},
};
use serde::{Deserialize, Serialize};
use sysinfo::{Disks, System};

pub const REPORT_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub version: u32,
    pub name: String,
    /// Address the other machines reach this one at (e.g. its Tailscale IP), if known.
    pub addr: Option<IpAddr>,
    pub os: String,
    pub cpu: String,
    pub cores: usize,
    pub ram_total: u64,
    /// Free space where model files will be stored.
    pub disk_free: u64,
    pub slots: Vec<Slot>,
    pub warnings: Vec<String>,
}

/// One device a node process can run on.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Slot {
    /// Value for `--device`: `cpu`, `cuda:N` or `metal`.
    pub device: String,
    pub label: String,
    /// Bytes this device may use for weights, KV cache and scratch.
    pub mem: u64,
    /// Measured decode speed, in GB of quantized weights streamed per second.
    pub gbps: f64,
}

impl Slot {
    /// Whether this slot lives in system memory, where the client's embedding and head go too.
    pub fn is_host(&self) -> bool {
        self.device == "cpu" || self.device == "metal"
    }
}

pub struct ProbeOpts {
    pub name: Option<String>,
    pub addr: Option<IpAddr>,
    pub models_dir: PathBuf,
    pub ram_limit: Option<u64>,
    pub vram_limit: Option<u64>,
}

pub fn probe(opts: &ProbeOpts) -> Result<Report> {
    let mut sys = System::new();
    sys.refresh_memory();
    sys.refresh_cpu_all();
    let cap = |bytes: u64, limit: Option<u64>| limit.map_or(bytes, |l| bytes.min(l));
    let ram_free = cap(sys.available_memory(), opts.ram_limit);
    let mut warnings = Vec::new();

    let mut slots = Vec::new();
    for g in nvidia_gpus() {
        if cfg!(feature = "cuda") {
            let mem = cap(g.free, opts.vram_limit);
            slots.push(Slot { device: format!("cuda:{}", g.index), label: g.name, mem, gbps: 0. });
        } else {
            warnings.push(format!("found {} but this build has no CUDA support; rebuild with --features cuda to use it", g.name));
        }
    }
    if cfg!(feature = "metal") {
        // Unified memory: the GPU and the CPU share RAM, and macOS lets the GPU wire about 3/4 of it.
        let mem = ram_free.min(sys.total_memory() / 4 * 3);
        slots.push(Slot { device: "metal".into(), label: "Apple GPU (unified memory)".into(), mem, gbps: 0. });
    } else {
        if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            warnings.push("this build has no Metal support; rebuild with --features metal to use the GPU".into());
        }
        let cpu = sys.cpus().first().map(|c| c.brand().trim().to_string()).unwrap_or_default();
        slots.push(Slot { device: "cpu".into(), label: cpu, mem: ram_free, gbps: 0. });
    }

    slots.retain_mut(|slot| match afo_engine::device(&slot.device).and_then(|d| bench(&d)) {
        Ok(gbps) => {
            slot.gbps = gbps;
            true
        }
        Err(e) => {
            warnings.push(format!("skipping {} ({}): {e:#}", slot.device, slot.label));
            false
        }
    });

    Ok(Report {
        version: REPORT_VERSION,
        name: opts.name.clone().or_else(System::host_name).unwrap_or_else(|| "unnamed".into()),
        addr: opts.addr.or_else(tailscale_ip),
        os: System::long_os_version().unwrap_or_default(),
        cpu: sys.cpus().first().map(|c| c.brand().trim().to_string()).unwrap_or_default(),
        cores: System::physical_core_count().unwrap_or(sys.cpus().len()),
        ram_total: sys.total_memory(),
        disk_free: disk_free(&opts.models_dir).unwrap_or(0),
        slots,
        warnings,
    })
}

/// (out, in) shapes of the matmuls in one Llama 3.1 8B block. At ~120 MB in Q4_K it is far bigger
/// than any cache, so like real decoding it measures how fast weights stream from memory.
const BENCH_SHAPES: [(usize, usize); 7] =
    [(4096, 4096), (1024, 4096), (1024, 4096), (4096, 4096), (14336, 4096), (14336, 4096), (4096, 14336)];

/// Times single-token forward passes through one 8B-sized block on `device`, including the
/// host round trip a node does every step, and returns GB of weights processed per second.
pub fn bench(device: &Device) -> Result<f64> {
    let mut bytes = 0;
    let mut mats = Vec::new();
    for (out, inp) in BENCH_SHAPES {
        let w = Tensor::randn(0f32, 1., (out, inp), &Device::Cpu)?;
        let q = QTensor::quantize_onto(&w, GgmlDType::Q4K, device)?;
        bytes += q.storage_size_in_bytes();
        mats.push(QMatMul::from_qtensor(q)?);
    }
    let [wq, wk, wv, wo, gate, up, down] = &mats[..] else { unreachable!() };
    let input = vec![0.01f32; 4096];
    let step = || -> Result<()> {
        let x = Tensor::from_slice(&input, (1, 4096), device)?;
        let att = (wo.forward(&x)? + wq.forward(&x)?)?;
        let (_k, _v) = (wk.forward(&x)?, wv.forward(&x)?);
        let h = (candle_nn::ops::silu(&gate.forward(&x)?)? * up.forward(&x)?)?;
        let y = (down.forward(&h)? + att)?;
        y.to_vec2::<f32>()?;
        Ok(())
    };

    for _ in 0..3 {
        step()?;
    }
    let (t, mut iters) = (Instant::now(), 0);
    while iters < 5 || (t.elapsed() < Duration::from_secs(1) && iters < 500) {
        step()?;
        iters += 1;
    }
    let secs = t.elapsed().as_secs_f64();
    ensure!(secs > 0., "benchmark finished too quickly to time");
    Ok(bytes as f64 * iters as f64 / secs / 1e9)
}

struct NvGpu {
    index: usize,
    name: String,
    free: u64,
}

fn nvidia_gpus() -> Vec<NvGpu> {
    let query = ["--query-gpu=index,name,memory.free", "--format=csv,noheader,nounits"];
    let Ok(out) = Command::new("nvidia-smi").args(query).output() else { return vec![] };
    if !out.status.success() {
        return vec![];
    }
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|line| {
            let fields: Vec<&str> = line.split(',').map(str::trim).collect();
            let [index, name, free] = fields[..] else { return None };
            Some(NvGpu { index: index.parse().ok()?, name: name.into(), free: free.parse::<u64>().ok()? << 20 })
        })
        .collect()
}

fn tailscale_ip() -> Option<IpAddr> {
    let out = Command::new("tailscale").args(["ip", "-4"]).output().ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8(out.stdout).ok()?.lines().next()?.trim().parse().ok()
}

/// Free space on the disk holding `dir`, or its nearest existing ancestor.
fn disk_free(dir: &Path) -> Option<u64> {
    let abs = std::path::absolute(dir).ok()?;
    let existing = abs.ancestors().find(|p| p.exists())?.canonicalize().ok()?;
    // Windows canonical paths are verbatim (`\\?\C:\...`) and never match a mount point like `C:\`.
    let existing = PathBuf::from(existing.to_string_lossy().trim_start_matches(r"\\?\").to_string());
    let disks = Disks::new_with_refreshed_list();
    disks
        .iter()
        .filter(|d| existing.starts_with(d.mount_point()))
        .max_by_key(|d| d.mount_point().as_os_str().len())
        .map(|d| d.available_space())
}
