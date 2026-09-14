#!/usr/bin/env bash
# Full comparison sweep. Each run is its own process so peak-memory numbers are isolated.
# VLLM_PYTHON lets the vLLM runs use a different interpreter (e.g. a venv that already has vLLM).
set -uo pipefail
MODEL=${MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}
INPUT=${INPUT:-256}
OUTPUT=${OUTPUT:-128}
NUM=${NUM:-64}
EXTRA=${EXTRA:-}
PY=${PYTHON:-python}
VPY=${VLLM_PYTHON:-$PY}
cd "$(dirname "$0")/.."
mkdir -p benchmarks/results
py_for() { [ "$1" = vllm ] && echo "$VPY" || echo "$PY"; }
for bs in 1 8 32 64; do
  for backend in hf pagedllm vllm; do
    echo "=== $backend batch=$bs ==="
    "$(py_for $backend)" -m benchmarks.bench --backend $backend --model "$MODEL" --batch-size $bs \
      --num-prompts $NUM --input-len $INPUT --output-len $OUTPUT $EXTRA || true
  done
done
# Prefix-caching scenario: 75% shared prefix.
for backend in pagedllm vllm; do
  "$(py_for $backend)" -m benchmarks.bench --backend $backend --model "$MODEL" --batch-size 32 --num-prompts $NUM \
    --input-len $INPUT --output-len $OUTPUT --shared-prefix 0.75 $EXTRA || true
done
"$PY" -m benchmarks.report "benchmarks/results/$(basename "$MODEL").jsonl" | tee benchmarks/results/latest.md
