"""Triton paged attention: reads K/V through the block table, no gather.

One program handles a BLOCK_M-row tile of one sequence's queries for one
head. It walks the sequence's context in BLOCK_N-row tiles; for every key
position n it looks up block_table[n // BLOCK_SIZE] and reads row
n % BLOCK_SIZE of that physical block. Online softmax (FlashAttention-2
style) keeps the running max / sum so the full [num_q, L] score matrix is
never materialised.

Prefill and decode share this kernel (a decode step is a 1-row query tile;
BLOCK_M is padded to 16 for tl.dot, which is why a dedicated split-K decode
kernel is the natural next optimisation).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_attention_kernel(
    Q, K, V, Out,
    BlockTables, CuSeqlensQ, SeqLensK,
    sm_scale,
    stride_qt, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_vb, stride_vs, stride_vh, stride_vd,
    stride_ot, stride_oh, stride_od,
    stride_bts, stride_btb,
    num_heads, num_kv_heads,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    q_tile = tl.program_id(0)
    head = tl.program_id(1)
    seq = tl.program_id(2)

    q_start = tl.load(CuSeqlensQ + seq)
    q_end = tl.load(CuSeqlensQ + seq + 1)
    num_q = q_end - q_start
    if q_tile * BLOCK_M >= num_q:
        return
    seq_len_k = tl.load(SeqLensK + seq)
    kv_head = head // (num_heads // num_kv_heads)

    offs_m = q_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_mask = offs_m < num_q
    q_ptrs = Q + (q_start + offs_m)[:, None] * stride_qt + head * stride_qh + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)

    # Absolute position of each query row; padding rows get positions >= seq_len_k
    # so they are never fully masked (avoids NaN) and are dropped at the store.
    q_pos = seq_len_k - num_q + offs_m

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Keys this tile can see: up to the last query row's position (causal).
    kv_end = tl.minimum(seq_len_k, seq_len_k - num_q + (q_tile + 1) * BLOCK_M)
    bt = BlockTables + seq * stride_bts

    for start_n in range(0, kv_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < kv_end
        phys = tl.load(bt + (offs_n // BLOCK_SIZE) * stride_btb, mask=n_mask, other=0).to(tl.int64)
        row = phys * stride_kb + (offs_n % BLOCK_SIZE) * stride_ks

        k_ptrs = K + row[:, None] + kv_head * stride_kh + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)          # [BLOCK_N, D]
        qk = tl.dot(q, tl.trans(k)) * sm_scale                         # [BLOCK_M, BLOCK_N]
        visible = (offs_n[None, :] <= q_pos[:, None]) & n_mask[None, :]
        qk = tl.where(visible, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)

        v_row = phys * stride_vb + (offs_n % BLOCK_SIZE) * stride_vs
        v_ptrs = V + v_row[:, None] + kv_head * stride_vh + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)          # [BLOCK_N, D]
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    o_ptrs = Out + (q_start + offs_m)[:, None] * stride_ot + head * stride_oh + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=q_mask[:, None])


def paged_attention_triton(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens_k: torch.Tensor,
    max_q_len: int,
    scale: float,
    block_m: int | None = None,
    block_n: int | None = None,
) -> torch.Tensor:
    assert q.dim() == 3 and k_cache.dim() == 4
    total_q, num_heads, head_dim = q.shape
    num_blocks, block_size, num_kv_heads, _ = k_cache.shape
    assert head_dim & (head_dim - 1) == 0, "head_dim must be a power of two"
    assert block_size & (block_size - 1) == 0, "block_size must be a power of two"
    num_seqs = seq_lens_k.shape[0]
    out = torch.empty_like(q)

    if block_m is None:
        block_m = 16 if max_q_len <= 16 else (64 if head_dim <= 64 else 32)
    if block_n is None:
        block_n = max(16, min(64, block_size))
    grid = (triton.cdiv(max_q_len, block_m), num_heads, num_seqs)
    _paged_attention_kernel[grid](
        q, k_cache, v_cache, out,
        block_tables, cu_seqlens_q, seq_lens_k,
        scale,
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        block_tables.stride(0), block_tables.stride(1),
        num_heads, num_kv_heads,
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=2,
    )
    return out
