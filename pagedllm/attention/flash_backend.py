"""flash-attn >= 2.6 varlen path with paged KV (`block_table=`)."""

from __future__ import annotations

import torch
from flash_attn import flash_attn_varlen_func


def paged_attention_flash(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens_k: torch.Tensor,
    max_q_len: int,
    scale: float,
) -> torch.Tensor:
    cu_seqlens_k = torch.zeros(seq_lens_k.shape[0] + 1, dtype=torch.int32, device=q.device)
    cu_seqlens_k[1:] = torch.cumsum(seq_lens_k, 0)
    return flash_attn_varlen_func(
        q,
        k_cache,
        v_cache,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_q_len,
        max_seqlen_k=int(seq_lens_k.max().item()),
        softmax_scale=scale,
        causal=True,
        block_table=block_tables,
    )
