"""Pure-PyTorch paged attention. Slow, obviously correct, runs on CPU.

Used as the oracle for the Triton kernel tests and as a fallback backend.
"""

from __future__ import annotations

import torch


def gather_kv(
    k_cache: torch.Tensor, v_cache: torch.Tensor, block_table: torch.Tensor, seq_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return contiguous [seq_len, num_kv_heads, head_dim] K and V for one sequence."""
    block_size = k_cache.shape[1]
    num_blocks = (seq_len + block_size - 1) // block_size
    blocks = block_table[:num_blocks].long()
    k = k_cache[blocks].reshape(-1, *k_cache.shape[2:])[:seq_len]
    v = v_cache[blocks].reshape(-1, *v_cache.shape[2:])[:seq_len]
    return k, v


def paged_attention_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens_k: torch.Tensor,
    max_q_len: int,
    scale: float,
) -> torch.Tensor:
    num_heads = q.shape[1]
    num_kv_heads = k_cache.shape[2]
    group = num_heads // num_kv_heads
    out = torch.empty_like(q)
    cu = cu_seqlens_q.tolist()
    lens = seq_lens_k.tolist()
    for s in range(len(lens)):
        q_start, q_end = cu[s], cu[s + 1]
        num_q = q_end - q_start
        L = lens[s]
        k, v = gather_kv(k_cache, v_cache, block_tables[s], L)  # [L, kvh, d]
        qs = q[q_start:q_end].float()                             # [nq, h, d]
        k = k.float().repeat_interleave(group, dim=1)             # [L, h, d]
        v = v.float().repeat_interleave(group, dim=1)
        scores = torch.einsum("qhd,khd->hqk", qs, k) * scale      # [h, nq, L]
        q_pos = torch.arange(L - num_q, L, device=q.device)
        k_pos = torch.arange(L, device=q.device)
        mask = k_pos[None, :] > q_pos[:, None]                    # [nq, L]
        scores = scores.masked_fill(mask[None], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        o = torch.einsum("hqk,khd->qhd", probs, v)
        out[q_start:q_end] = o.to(q.dtype)
    return out
