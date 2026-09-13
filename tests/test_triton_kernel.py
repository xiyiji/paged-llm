"""GPU-only: Triton paged attention vs the PyTorch reference."""
import math

import pytest
import torch

from tests.test_attention_reference import build_case

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from pagedllm.attention.reference import paged_attention_reference  # noqa: E402
from pagedllm.attention.triton_kernel import paged_attention_triton  # noqa: E402


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("block_size", [16, 32])
@pytest.mark.parametrize("num_heads,num_kv_heads,head_dim", [(8, 8, 64), (32, 8, 128), (32, 4, 64)])
@pytest.mark.parametrize("seq_lens_k,num_q", [
    ([7], [7]),
    ([129, 300, 17], [129, 300, 17]),        # prefill, crosses many blocks
    ([1000, 33, 2048, 5], [1, 1, 1, 1]),     # decode
    ([500, 64, 77], [37, 1, 12]),            # mixed / chunked
    ([1], [1]),                              # first token of a sequence
])
def test_triton_matches_reference(dtype, block_size, num_heads, num_kv_heads, head_dim, seq_lens_k, num_q):
    q, kc, vc, bt, cu, sl, max_q = build_case(seq_lens_k, num_q, num_heads, num_kv_heads, head_dim,
                                              block_size, dtype=dtype, device="cuda", seed=1)
    scale = 1 / math.sqrt(head_dim)
    want = paged_attention_reference(q, kc, vc, bt, cu, sl, max_q, scale)
    got = paged_attention_triton(q, kc, vc, bt, cu, sl, max_q, scale)
    tol = 2e-2 if dtype == torch.bfloat16 else 1e-2
    torch.testing.assert_close(got.float(), want.float(), atol=tol, rtol=tol)


def test_triton_block_table_padding_is_ignored():
    # Garbage in padded block-table entries must not leak into results.
    q, kc, vc, bt, cu, sl, max_q = build_case([20, 5], [1, 1], block_size=16, dtype=torch.float16, device="cuda")
    bt2 = bt.clone(); bt2[1, 1:] = 999_999  # out-of-range but never dereferenced
    scale = 1 / math.sqrt(q.shape[-1])
    a = paged_attention_triton(q, kc, vc, bt, cu, sl, max_q, scale)
    b = paged_attention_triton(q, kc, vc, bt2, cu, sl, max_q, scale)
    torch.testing.assert_close(a, b)
