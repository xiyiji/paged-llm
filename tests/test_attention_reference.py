"""The reference paged attention must equal dense causal attention on the gathered KV."""
import math

import pytest
import torch

from pagedllm.attention.reference import paged_attention_reference


def build_case(seq_lens_k, num_q, num_heads=4, num_kv_heads=2, head_dim=32, block_size=8, dtype=torch.float32, device="cpu", seed=0):
    g = torch.Generator().manual_seed(seed)
    num_seqs = len(seq_lens_k)
    max_blocks = max((L + block_size - 1) // block_size for L in seq_lens_k)
    total_blocks = sum((L + block_size - 1) // block_size for L in seq_lens_k) + 3
    perm = torch.randperm(total_blocks, generator=g)
    block_tables = torch.zeros(num_seqs, max_blocks, dtype=torch.int32)
    p = 0
    for s, L in enumerate(seq_lens_k):
        nb = (L + block_size - 1) // block_size
        block_tables[s, :nb] = perm[p:p + nb]
        p += nb
    k_cache = torch.randn(total_blocks, block_size, num_kv_heads, head_dim, generator=g).to(dtype)
    v_cache = torch.randn(total_blocks, block_size, num_kv_heads, head_dim, generator=g).to(dtype)
    q = torch.randn(sum(num_q), num_heads, head_dim, generator=g).to(dtype)
    cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(num_q), 0)), dtype=torch.int32)
    seq_lens = torch.tensor(seq_lens_k, dtype=torch.int32)
    to = lambda t: t.to(device)
    return to(q), to(k_cache), to(v_cache), to(block_tables), to(cu), to(seq_lens), max(num_q)


def dense_reference(q, k_cache, v_cache, block_tables, cu, seq_lens, scale):
    """Independent implementation: gather to contiguous tensors, use torch SDPA with an explicit mask."""
    block_size = k_cache.shape[1]
    group = q.shape[1] // k_cache.shape[2]
    out = torch.empty_like(q)
    for s in range(len(seq_lens)):
        L = int(seq_lens[s]); nq = int(cu[s + 1] - cu[s])
        idx = torch.arange(L)
        rows = block_tables[s][idx // block_size].long() * block_size + idx % block_size
        k = k_cache.reshape(-1, *k_cache.shape[2:])[rows].repeat_interleave(group, 1).float()
        v = v_cache.reshape(-1, *v_cache.shape[2:])[rows].repeat_interleave(group, 1).float()
        qs = q[cu[s]:cu[s + 1]].float()
        mask = torch.ones(nq, L, dtype=torch.bool).tril(diagonal=L - nq)
        o = torch.nn.functional.scaled_dot_product_attention(
            qs.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), attn_mask=mask, scale=scale)
        out[cu[s]:cu[s + 1]] = o.transpose(0, 1).to(q.dtype)
    return out


@pytest.mark.parametrize("seq_lens_k,num_q", [
    ([5], [5]),                # single prefill, partial block
    ([33, 8, 17], [33, 8, 17]),  # batched prefill
    ([40, 9, 64], [1, 1, 1]),  # decode
    ([40, 9, 64], [8, 1, 3]),  # chunked prefill + decode mixed
])
def test_reference_matches_dense(seq_lens_k, num_q):
    q, kc, vc, bt, cu, sl, max_q = build_case(seq_lens_k, num_q)
    scale = 1 / math.sqrt(q.shape[-1])
    got = paged_attention_reference(q, kc, vc, bt, cu, sl, max_q, scale)
    want = dense_reference(q, kc, vc, bt, cu, sl, scale)
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)
