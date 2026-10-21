# AllForOne

Run a model too big for any one machine by splitting its layers across several people's
computers over the internet.

```
client ──► node 1 (layers 0..a) ──► node 2 (a..b) ──► … ──► node N ──┐
  ▲  embedding + output head stay here                               │
  └──────────────────────────── hidden states ◄──────────────────────┘
```

- **Pipeline parallelism.** Each node loads only its block of layers from a GGUF file and keeps
  the KV cache for those layers.
- **Push-forward ring.** Each node sends straight to the next one, so a token costs N one-way
  hops instead of N round trips.
- **The client keeps the ends of the model.** Token ids and logits never leave the user's machine.
- **QUIC transport** (`quinn`). One persistent stream per hop with length-prefixed frames,
  TLS 1.3 encryption, and large flow-control and initial congestion windows for prefill bursts.
- **Swappable activation codecs** (`f32`, `f16`, `fp8`, `int8`, with a per-token scale). This is
  also where the security phase will plug in (for example RNS sharing).

## Crates

| crate | role |
|---|---|
| `afo-engine` | Layer-sliced quantized Llama (candle), `Engine` trait, per-session KV cache |
| `afo-codec` | `ActivationCodec` implementations |
| `afo-proto` | Wire messages and the QUIC transport (`Net`) |
| `afo-node` | Node binary: runs one block of layers |
| `afo-client` | Client library and CLI: embedding, head, sampling, `LocalPipeline` / `RemotePipeline` |
| `afo-bench` | Checks split execution against a single-machine reference and reports latency |
| `afo-plan` | `afo-probe` measures a machine; `afo-plan` picks a model and layer split for the group |
| `afo-monitor` | Web dashboard: which nodes are up, what they run, their resource use, tokens/s |

## Quickstart (one machine)

```sh
mkdir -p models
curl -L -o models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf
curl -L -o models/tokenizer.json \
  https://huggingface.co/unsloth/Llama-3.2-1B-Instruct/resolve/main/tokenizer.json

scripts/local-ring.sh                    # 3 nodes on localhost + correctness/latency report
SIM_DELAY_MS=30 scripts/local-ring.sh    # emulate a 30 ms one-way WAN hop
```

## Across machines

```sh
# on each node machine (layers are half-open; the ranges must cover 0..n_layer in order)
afo-node --model model.gguf --layers 0..40  --listen 0.0.0.0:7100      # machine A
afo-node --model model.gguf --layers 40..80 --listen 0.0.0.0:7100      # machine B

# on the user's machine; --advertise is the IP that the last node can reach
afo-client --model model.gguf --tokenizer tokenizer.json \
  --route A:7100,B:7100 --advertise <my-ip> --prompt "Hello"
```

Build with `--features cuda` or `--features metal` on `afo-node` or `afo-client` to use a GPU,
then pass `--device cuda:0` or `--device metal`.

**The CUDA toolkit must not be newer than the driver.** `nvidia-smi` shows the newest CUDA version
the driver supports. Kernels compiled by a newer `nvcc` fail at load time with
`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`. Point the build at a toolkit no newer than that, and give
the host compiler that toolkit supports (CUDA 12.0 needs gcc 12 or older):
```sh
export PATH=/usr/bin:$PATH        # where the older nvcc lives
export NVCC_CCBIN=/usr/bin/gcc-12 CUDA_COMPUTE_CAP=86   # compute_cap from nvidia-smi --query-gpu=compute_cap --format=csv
cargo clean --release -p candle-kernels -p cudarc -p candle-core   # after switching toolkits
cargo build --release -p afo-node -p afo-client -p afo-plan -p afo-bench \
  --features afo-node/cuda,afo-client/cuda,afo-plan/cuda
```
Results on a GPU match the CPU closely but not bit for bit. On Llama 3.2 1B, greedy output
matched the CPU reference for 80 tokens before a near-tie went the other way.
Each address must be reachable directly for now. NAT traversal comes in Phase 3.

## Windows

Everything runs natively on Windows 10 and 11 without WSL. `afo-plan` sees from each probe report
which machines run Windows and prints PowerShell commands for them.

**Build on the Windows machine:**
1. Install [Rust](https://rustup.rs) with the default MSVC toolchain. Its installer offers the
   Visual Studio Build Tools; accept them.
2. CPU only: in PowerShell, `cargo build --release`. The programs are in `target\release\`.
3. NVIDIA GPU: install the CUDA toolkit for Windows (no newer than the CUDA version `nvidia-smi`
   shows; see above), then build from the **x64 Native Tools Command Prompt for VS** so `nvcc`
   finds `cl.exe`:
   ```bat
   set CUDA_COMPUTE_CAP=86
   cargo build --release -p afo-node -p afo-client -p afo-plan --features afo-node/cuda,afo-client/cuda,afo-plan/cuda
   ```

**Or cross-compile a CPU-only build from Linux** and copy the `.exe` files over. It needs only
DLLs that ship with Windows:
```sh
rustup target add x86_64-pc-windows-gnu
pip install ziglang && cargo install cargo-zigbuild
cargo zigbuild --release --target x86_64-pc-windows-gnu
```

**Allow the ports through Windows Firewall** (PowerShell as Administrator). Nodes use UDP for the
model traffic and TCP for the monitor's status checks. Add 7200 if this machine runs the client.
```powershell
New-NetFirewallRule -DisplayName "AllForOne UDP" -Direction Inbound -Protocol UDP -LocalPort 7100-7110,7200 -Action Allow -Profile Private,Domain
New-NetFirewallRule -DisplayName "AllForOne TCP" -Direction Inbound -Protocol TCP -LocalPort 7100-7110 -Action Allow -Profile Private,Domain
```
If the network is set to "Public", either switch it to Private or add `Public` to `-Profile`.

## Planning a group

Instead of choosing a model and layer ranges by hand, let the planner do it from measurements.

1. **Every machine, including yours**, builds and runs the probe. Use the same GPU feature you'll run
   nodes with: `--features cuda` on NVIDIA, `--features metal` on Apple silicon.
   ```sh
   cargo build --release -p afo-plan -p afo-node -p afo-client --features afo-node/cuda,afo-plan/cuda
   target/release/afo-probe --name sara      # writes sara.json
   ```
   It reports usable RAM and VRAM, free disk under `models/`, and benchmarks each device for a few
   seconds. It picks up a Tailscale IP automatically. Otherwise pass `--addr <ip>`. Pass
   `--ram-limit-gb` or `--vram-limit-gb` to offer less than everything.
2. **Everyone sends their JSON** to whoever runs the client.
3. **The client runs the planner** on all of the reports:
   ```sh
   target/release/afo-plan ahmed.json sara.json omar.json --client ahmed
   ```
   It lists every catalog model as fitting or not, with an estimated speed. It chooses the best
   model that fits and reaches `--min-tok-s` (default 2), then prints each person's download and
   `afo-node` command and the client's `afo-client` command.

How the planner decides:
- **Best model.** More parameters first, then more bits per weight. The catalog has Llama 3.2
  1B/3B, Llama 3.1 8B and Llama 3.3 70B in single-file GGUFs (`crates/plan/src/catalog.rs`).
- **Memory.** Each device needs room for its layers' weights, an f32 KV cache for `--ctx` tokens
  (default 4096), prefill scratch for a `--prompt-tokens` prompt (default 1024) and runtime
  overhead. The client's machine also holds the f32 embedding and the output head. Each machine
  that runs anything needs the whole model file on disk.
- **Speed.** Per token, every layer runs once in sequence, so time is the sum of each layer's time on
  its device plus `--hop-ms` (default 30) per network hop. The planner fills the fastest devices
  first, and leaves out machines whose extra hop costs more than the layers they would take. A
  machine with a GPU can run two nodes, one on the GPU and one on its CPU, on consecutive ports.
- The probe benchmarks one Llama-8B-sized block of Q4_K weights. On the 1B model it predicted
  16.8 tok/s against a measured 17.2 (CPU, one thread per physical core). CPU commands set
  `RAYON_NUM_THREADS` to the physical core count, because extra hyperthreads slow decoding down.

## Monitoring

Every node answers `GET /status` over **TCP** on the same port number it uses for QUIC (UDP), so
nothing else needs opening. `afo-monitor` polls the nodes once a second and serves a dashboard:

```sh
afo-monitor --nodes 100.64.0.2:7100,100.64.0.3:7100        # the same list as --route
afo-plan *.json --json > plan.json && afo-monitor --plan plan.json
```

Open http://127.0.0.1:7300. Add `--listen 0.0.0.0:7300` to view it from another device, such as
your phone over Tailscale. The page shows:
- ring health, the model, and how many layers live nodes cover
- tokens/s now, the peak, and a 5-minute chart. Every generated token passes every node, so a node's
  step rate is the ring's generation rate. The client doesn't need to report anything.
- a layer map showing which node runs which layers, with missing layers hatched
- per node: online or offline, CPU, RAM, GPU and VRAM use (NVIDIA via `nvidia-smi`), busy share,
  last step time, step and token counts

The status endpoint is read-only but unauthenticated. It reveals the model, hostname and
resource use to anyone who can reach the port.

## Phase 1 results (Llama-3.2-1B Q4_K_M, 3 local nodes, CPU, 4 threads each)

| run | tokens matching reference | max \|Δlogit\| | decode tok/s | network ms/token |
|---|---|---|---|---|
| reference (single process) | 40/40 | 0 | 25.1 | 0 |
| ring, f32 | 40/40 | 0 (bit-exact) | 19.3 | 0.86 |
| ring, f16 | 40/40 | 0.86 | 17.7 | 0.87 |
| ring, fp8 | 40/40 | 1.19 | 19.3 | 0.84 |
| ring, int8 | 40/40 | 1.76 | 19.3 | 0.79 |
| ring, f16, 30 ms/hop simulated | 24/24 | 0.82 | 5.4 | 127.8 |

Over a simulated WAN, decoding is limited by the network: 4 hops × 30 ms caps it at about 8
tokens/s before any compute. Phase 2 (speculative decoding) targets this.

## Roadmap

1. ✅ Pipeline over QUIC with fixed routes; correctness checked against the reference.
2. Speculative decoding (a local draft model; a whole draft block verified in one ring pass) and
   prefill overlap.
3. Coordinator: discovery, latency-aware layer placement, NAT traversal, failover with KV replay.
   Offline planning from hardware probes (`afo-plan`) is done. Live discovery is next.
4. Security: node identity, integrity spot-checks, an RNS research codec for private linear blocks.

Known Phase 1 limits:
- Only the `llama` GGUF architecture is supported.
- TLS certificates are not verified (traffic is encrypted, but nodes are not authenticated).
- The KV cache grows by concatenation.
- One session is computed at a time per node.
