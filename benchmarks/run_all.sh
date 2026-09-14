#!/usr/bin/env bash
# Full comparison sweep. Each run is its own process so peak-memory numbers are isolated.
# VLLM_PYTHON lets the vLLM runs use a different interpreter (e.g. a venv that already has vLLM);
# its bin dir is put on PATH so vLLM finds ninja etc. BACKENDS / BATCHES select a subset.
set -uo pipefail
MODEL=${MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}
INPUT=${INPUT:-256}
OUTPUT=${OUTPUT:-128}
NUM=${NUM:-64}
EXTRA=${EXTRA:-}
PY=${PYTHON:-python}
VPY=${VLLM_PYTHON:-$PY}
BACKENDS=${BACKENDS:-hf pagedllm vllm}
BATCHES=${BATCHES:-1 8 32 64}
cd "$(dirname "$0")/.."
mkdir -p benchmarks/results
py_for() { [ "$1" = vllm ] && echo "$VPY" || echo "$PY"; }
run_one() { local backend=$1; shift; PATH="$(dirname "$(py_for "$backend")"):$PATH" "$(py_for "$backend")" -m benchmarks.bench --backend "$backend" "$@" || true; }
for bs in $BATCHES; do
  for backend in $BACKENDS; do
    echo "=== $backend batch=$bs ==="
    run_one "$backend" --model "$MODEL" --batch-size $bs --num-prompts $NUM --input-len $INPUT --output-len $OUTPUT $EXTRA
  done
done
# Ragged workload: random input/output lengths (where static batching pads and waits).
for bs in 32 64; do
  for backend in $BACKENDS; do
    echo "=== $backend batch=$bs var-len ==="
    run_one "$backend" --model "$MODEL" --batch-size $bs --num-prompts $NUM --input-len $INPUT --output-len $OUTPUT --var-len $EXTRA
  done
done
# Prefix-caching scenario: 75% shared prefix.
for backend in $BACKENDS; do
  [ "$backend" = hf ] && continue
  run_one "$backend" --model "$MODEL" --batch-size 32 --num-prompts $NUM --input-len $INPUT --output-len $OUTPUT --shared-prefix 0.75 $EXTRA
done
"$PY" -m benchmarks.report "benchmarks/results/$(basename "$MODEL").jsonl" | tee benchmarks/results/latest.md
