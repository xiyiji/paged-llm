#!/usr/bin/env bash
# Full comparison sweep. Each run is its own process so peak-memory numbers are isolated.
set -uo pipefail
MODEL=${MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}
INPUT=${INPUT:-256}
OUTPUT=${OUTPUT:-128}
NUM=${NUM:-64}
EXTRA=${EXTRA:-}
cd "$(dirname "$0")/.."
for bs in 1 8 32 64; do
  for backend in hf pagedllm vllm; do
    echo "=== $backend batch=$bs ==="
    python -m benchmarks.bench --backend $backend --model "$MODEL" --batch-size $bs \
      --num-prompts $NUM --input-len $INPUT --output-len $OUTPUT $EXTRA || true
  done
done
# Prefix-caching scenario: 75% shared prefix.
for backend in pagedllm vllm; do
  python -m benchmarks.bench --backend $backend --model "$MODEL" --batch-size 32 --num-prompts $NUM \
    --input-len $INPUT --output-len $OUTPUT --shared-prefix 0.75 $EXTRA || true
done
python -m benchmarks.report "benchmarks/results/$(basename "$MODEL").jsonl" | tee benchmarks/results/latest.md
