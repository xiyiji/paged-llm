#!/usr/bin/env bash
# Bootstrap on a fresh RunPod "PyTorch 2.x + CUDA 12.x" pod, then run tests + benchmarks.
#   git clone <your fork> pagedllm && cd pagedllm && bash scripts/runpod_setup.sh
# Optional: export HF_TOKEN=... for gated models (Llama-3).
set -euo pipefail
cd "$(dirname "$0")/.."
pip install -q -U pip uv
# vLLM pins its own torch/triton; install it first so everything else follows its versions.
uv pip install --system -q vllm pynvml tqdm pytest
uv pip install --system -q -e .
# flash-attn is optional (prebuilt wheel install can take a while); skip with SKIP_FLASH=1
if [ -z "${SKIP_FLASH:-}" ]; then
  uv pip install --system -q flash-attn --no-build-isolation || echo "flash-attn install failed; continuing with Triton backend only"
fi
python - <<'PY'
import torch, triton
print("torch", torch.__version__, "triton", triton.__version__, "gpu", torch.cuda.get_device_name(0))
PY
echo "=== unit tests (CPU logic + Triton kernel vs reference) ==="
pytest -q tests/test_block_manager.py tests/test_scheduler.py tests/test_attention_reference.py tests/test_engine_cpu.py tests/test_triton_kernel.py
echo "=== real model tests ==="
pytest -q tests/test_model_gpu.py
echo "=== kernel micro-benchmark ==="
python -m benchmarks.bench_kernel | tee benchmarks/results/kernel.md
echo "=== end-to-end benchmark sweep ==="
bash benchmarks/run_all.sh
