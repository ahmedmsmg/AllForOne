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
Each address must be reachable directly for now. NAT traversal comes in Phase 3.

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
4. Security: node identity, integrity spot-checks, an RNS research codec for private linear blocks.

Known Phase 1 limits:
- Only the `llama` GGUF architecture is supported.
- TLS certificates are not verified (traffic is encrypted, but nodes are not authenticated).
- The KV cache grows by concatenation.
- One session is computed at a time per node.
