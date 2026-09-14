#!/usr/bin/env bash
# Everything that needs a GPU, in order, with logs under benchmarks/results/.
#   HF_HOME=/workspace/hf VLLM_PYTHON=/path/to/venv/bin/python bash scripts/gpu_run.sh
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p benchmarks/results
LOG=benchmarks/results/gpu_run.log
{
  echo "### $(date -u +%FT%TZ) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
  echo "=== 1/4 Triton kernel vs reference ==="
  python -m pytest -q tests/test_triton_kernel.py -p no:cacheprovider 2>&1 | tail -3
  echo "=== 2/4 TinyLlama parity with HF ==="
  python -m pytest -q tests/test_model_gpu.py -p no:cacheprovider 2>&1 | grep -vE "Warning|pynvml" | tail -15
  echo "=== 3/4 kernel micro-benchmark ==="
  python -m benchmarks.bench_kernel 2>&1 | grep -vE "Warning|pynvml" | tee benchmarks/results/kernel.md
  echo "=== 4/4 end-to-end HF / pagedllm / vLLM ==="
  bash benchmarks/run_all.sh 2>&1 | grep -vE "Warning|pynvml|it/s\]"
  echo "### done $(date -u +%FT%TZ)"
} 2>&1 | tee -a "$LOG"
