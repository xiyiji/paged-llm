"""Micro-benchmark of the attention backends on synthetic paged KV.

    python -m benchmarks.bench_kernel            # triton vs flash (if installed) vs torch reference (small shapes only)
"""
from __future__ import annotations

import math

import torch
import triton

from pagedllm.attention import _has_flash
from pagedllm.attention.reference import paged_attention_reference
from pagedllm.attention.triton_kernel import paged_attention_triton
from tests.test_attention_reference import build_case


def bench(fn, *a):
    return triton.testing.do_bench(lambda: fn(*a), warmup=10, rep=50)


def main():
    heads, kv_heads, hd, bs = 32, 8, 128, 16   # Llama-3-8B shape
    backends = {"triton": paged_attention_triton}
    if _has_flash():
        from pagedllm.attention.flash_backend import paged_attention_flash
        backends["flash-attn"] = paged_attention_flash
    print(f"GPU: {torch.cuda.get_device_name(0)}  heads={heads} kv_heads={kv_heads} head_dim={hd} block_size={bs}\n")
    print("| workload | batch | context | " + " | ".join(f"{b} (us)" for b in backends) + " | torch ref (us) |")
    print("|---|---:|---:|" + "---:|" * (len(backends) + 1))
    cases = [("decode", b, ctx, [1] * b, [ctx] * b) for b in (1, 8, 32, 64) for ctx in (512, 2048, 8192)]
    cases += [("prefill", 1, n, [n], [n]) for n in (512, 2048)]
    cases += [("chunked prefill (512 new)", 4, 2048, [512] * 4, [2048] * 4)]
    for name, b, ctx, nq, lens in cases:
        q, kc, vc, bt, cu, sl, max_q = build_case(lens, nq, heads, kv_heads, hd, bs, dtype=torch.float16, device="cuda")
        scale = 1 / math.sqrt(hd)
        cols = [f"{bench(fn, q, kc, vc, bt, cu, sl, max_q, scale) * 1000:.0f}" for fn in backends.values()]
        ref = f"{bench(paged_attention_reference, q, kc, vc, bt, cu, sl, max_q, scale) * 1000:.0f}" if b * ctx <= 8192 else "-"
        print(f"| {name} | {b} | {ctx} | " + " | ".join(cols) + f" | {ref} |")


if __name__ == "__main__":
    main()
