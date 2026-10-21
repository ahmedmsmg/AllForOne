#!/usr/bin/env bash
# Starts a 3-node ring on localhost and runs the correctness/latency benchmark against it.
#   SIM_DELAY_MS=30 scripts/local-ring.sh     # emulate 30 ms one-way per hop
# Extra arguments are passed to afo-bench.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-models/Llama-3.2-1B-Instruct-Q4_K_M.gguf}
TOKENIZER=${TOKENIZER:-models/tokenizer.json}
SPLITS=${SPLITS:-"0..6 6..11 11..16"}
SIM_DELAY_MS=${SIM_DELAY_MS:-0}
THREADS=${THREADS:-4}
BIN=target/release

cargo build --release -q
LOGDIR=$(mktemp -d)
pids=()
trap 'kill "${pids[@]}" 2>/dev/null || true' EXIT

port=7101
route=()
for layers in $SPLITS; do
  RAYON_NUM_THREADS=$THREADS $BIN/afo-node --model "$MODEL" --layers "$layers" \
    --listen 127.0.0.1:$port --sim-delay-ms "$SIM_DELAY_MS" >"$LOGDIR/node-$port.log" 2>&1 &
  pids+=($!)
  route+=(127.0.0.1:$port)
  port=$((port + 1))
done

for log in "$LOGDIR"/node-*.log; do
  until grep -q listening "$log"; do
    if ! kill -0 "${pids[@]}" 2>/dev/null; then cat "$LOGDIR"/*.log; exit 1; fi
    sleep 0.2
  done
done
echo "ring up: ${route[*]} (logs in $LOGDIR)"

RAYON_NUM_THREADS=$THREADS $BIN/afo-bench --model "$MODEL" --tokenizer "$TOKENIZER" \
  --route "$(IFS=,; echo "${route[*]}")" --sim-delay-ms "$SIM_DELAY_MS" "$@"
