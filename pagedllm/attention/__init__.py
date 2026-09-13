"""Attention backends that read K/V straight from the paged cache via a block table.

All backends share one signature:

    out = paged_attention(q, k_cache, v_cache, block_tables, cu_seqlens_q,
                          seq_lens_k, max_q_len, scale)

    q            [total_q, num_heads, head_dim]   new query tokens, all sequences concatenated
    k_cache      [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache      same
    block_tables [num_seqs, max_blocks] int32, padded with 0
    cu_seqlens_q [num_seqs + 1] int32           query offsets
    seq_lens_k   [num_seqs] int32               context length INCLUDING the new tokens
    Returns      [total_q, num_heads, head_dim]

Causal semantics: query i of sequence s sits at absolute position
seq_lens_k[s] - num_q[s] + i and attends to keys 0..that position.
Prefill (num_q = prompt chunk) and decode (num_q = 1) use the same call.
"""

from __future__ import annotations

import torch

from pagedllm.attention.reference import paged_attention_reference


def _has_triton() -> bool:
    try:
        import triton  # noqa: F401
        return torch.cuda.is_available()
    except ImportError:
        return False


def _has_flash() -> bool:
    try:
        import flash_attn  # noqa: F401
        return torch.cuda.is_available()
    except ImportError:
        return False


def select_backend(name: str = "auto"):
    if name == "auto":
        name = "triton" if _has_triton() else ("flash" if _has_flash() else "torch")
    if name == "triton":
        from pagedllm.attention.triton_kernel import paged_attention_triton
        return paged_attention_triton
    if name == "flash":
        from pagedllm.attention.flash_backend import paged_attention_flash
        return paged_attention_flash
    if name == "torch":
        return paged_attention_reference
    raise ValueError(f"unknown attention backend {name!r}")
