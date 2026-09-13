"""Physical KV storage: one K and one V tensor per layer.

Layout: [num_blocks, block_size, num_kv_heads, head_dim].
Flattening the first two dims gives a "slot" index = block_id * block_size + offset,
so writing new tokens is a single index_copy_. This is also the layout
flash-attn's paged KV path (`block_table=`) expects.
"""

from __future__ import annotations

import torch

from pagedllm.block_manager import CopyOnWrite


class KVCache:
    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ):
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)
        shape = (num_blocks, block_size, num_kv_heads, head_dim)
        self.k_caches = [torch.zeros(shape, dtype=dtype, device=self.device) for _ in range(num_layers)]
        self.v_caches = [torch.zeros(shape, dtype=dtype, device=self.device) for _ in range(num_layers)]

    @staticmethod
    def block_bytes(num_layers: int, block_size: int, num_kv_heads: int, head_dim: int, dtype: torch.dtype) -> int:
        return 2 * num_layers * block_size * num_kv_heads * head_dim * torch.tensor([], dtype=dtype).element_size()

    def write(self, layer: int, k: torch.Tensor, v: torch.Tensor, slot_mapping: torch.Tensor) -> None:
        """k, v: [num_tokens, num_kv_heads, head_dim]; slot_mapping: [num_tokens] int64."""
        self.k_caches[layer].view(-1, self.num_kv_heads, self.head_dim).index_copy_(0, slot_mapping, k)
        self.v_caches[layer].view(-1, self.num_kv_heads, self.head_dim).index_copy_(0, slot_mapping, v)

    def apply_copies(self, ops: list[CopyOnWrite]) -> None:
        if not ops:
            return
        src = torch.tensor([o.src_block for o in ops], device=self.device)
        dst = torch.tensor([o.dst_block for o in ops], device=self.device)
        for layer in range(self.num_layers):
            self.k_caches[layer][dst] = self.k_caches[layer][src]
            self.v_caches[layer][dst] = self.v_caches[layer][src]
