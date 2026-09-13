"""Paged KV-cache block management with hash-based prefix caching.

Design (mirrors vLLM v1's KVCacheManager, simplified):

* The KV cache is a pool of `num_blocks` fixed-size blocks. Each block holds
  `block_size` token slots for every layer. Physical storage lives in
  `KVCache` (see kv_cache.py); this module only tracks *which* block a
  sequence's tokens live in.
* Every block has a reference count. A block can be shared by several
  sequences (prefix caching, fork). It is returned to the free pool when
  the count drops to zero.
* Free blocks are kept in an LRU-ordered queue. A freed block keeps its
  content hash so that a later request with the same prefix can re-use it
  ("cached but evictable"). Allocating a free block that still has a hash
  evicts that cache entry.
* Full blocks are hashed as hash(prev_block_hash, tokens_in_block); the
  chain makes the hash depend on the whole prefix, not just 16 tokens.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

from pagedllm.sequence import Sequence


@dataclass
class KVBlock:
    block_id: int
    ref_count: int = 0
    block_hash: int | None = None


@dataclass
class CopyOnWrite:
    """Physical copy the model runner must perform before the next forward."""

    src_block: int
    dst_block: int


def _hash_block(prev_hash: int | None, tokens: list[int]) -> int:
    # Python's tuple hash is stable within a process, which is all we need.
    return hash((prev_hash, tuple(tokens)))


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int, enable_prefix_caching: bool = True):
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        self.blocks = [KVBlock(i) for i in range(num_blocks)]
        # LRU free queue: head = least recently freed (evict first).
        self._free: OrderedDict[int, None] = OrderedDict((i, None) for i in range(num_blocks))
        # content hash -> block id, only for blocks whose content is complete.
        self._cached: dict[int, int] = {}
        self._pending_cow: list[CopyOnWrite] = []
        # stats
        self.num_cache_queries = 0
        self.num_cache_hits = 0

    # ---- introspection ---------------------------------------------------
    @property
    def num_free_blocks(self) -> int:
        return len(self._free)

    @property
    def num_cached_blocks(self) -> int:
        return len(self._cached)

    def usage(self) -> float:
        return 1.0 - self.num_free_blocks / self.num_blocks

    @staticmethod
    def num_blocks_for_tokens(num_tokens: int, block_size: int) -> int:
        return (num_tokens + block_size - 1) // block_size

    # ---- low-level pool ops --------------------------------------------------
    def _allocate_block(self) -> KVBlock:
        block_id, _ = self._free.popitem(last=False)  # LRU head
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.block_hash is not None:
            # Evict from prefix cache: this physical block is being reused.
            if self._cached.get(block.block_hash) == block_id:
                del self._cached[block.block_hash]
            block.block_hash = None
        block.ref_count = 1
        return block

    def _touch(self, block_id: int) -> None:
        """Take a reference on a block, removing it from the free queue if needed."""
        block = self.blocks[block_id]
        if block.ref_count == 0:
            del self._free[block_id]
        block.ref_count += 1

    def _release(self, block_id: int) -> None:
        block = self.blocks[block_id]
        assert block.ref_count > 0, f"double free of block {block_id}"
        block.ref_count -= 1
        if block.ref_count == 0:
            if not self.enable_prefix_caching:
                block.block_hash = None
            self._free[block_id] = None  # MRU tail

    # ---- prefix cache --------------------------------------------------------
    def get_computed_blocks(self, seq: Sequence) -> tuple[list[int], int]:
        """Longest cached prefix of `seq.token_ids` in whole blocks.

        Returns (block_ids, num_cached_tokens). At least one token is always
        left uncomputed so the forward pass produces logits for sampling.
        """
        if not self.enable_prefix_caching or seq.num_computed_tokens > 0:
            return [], 0
        self.num_cache_queries += 1
        hits: list[int] = []
        prev_hash: int | None = None
        num_full = (seq.num_tokens - 1) // self.block_size  # never cache-hit the last token
        for i in range(num_full):
            tokens = seq.token_ids[i * self.block_size:(i + 1) * self.block_size]
            h = _hash_block(prev_hash, tokens)
            block_id = self._cached.get(h)
            if block_id is None:
                break
            hits.append(block_id)
            prev_hash = h
        if hits:
            self.num_cache_hits += 1
        return hits, len(hits) * self.block_size

    def cache_full_blocks(self, seq: Sequence, num_computed_before: int, num_computed_after: int) -> None:
        """Register blocks that became full between the two watermarks."""
        if not self.enable_prefix_caching:
            return
        bs = self.block_size
        first_full = num_computed_before // bs
        last_full = num_computed_after // bs  # exclusive
        if first_full == last_full:
            return
        prev_hash = self.blocks[seq.block_table[first_full - 1]].block_hash if first_full > 0 else None
        for i in range(first_full, last_full):
            block = self.blocks[seq.block_table[i]]
            if block.block_hash is None:
                if prev_hash is None and i > 0:
                    # Predecessor un-hashed (e.g. duplicate content computed concurrently);
                    # cannot chain further.
                    return
                h = _hash_block(prev_hash, seq.token_ids[i * bs:(i + 1) * bs])
                if h in self._cached:
                    # Same content already cached in a different block; leave this one unhashed.
                    return
                block.block_hash = h
                self._cached[h] = block.block_id
            prev_hash = block.block_hash

    # ---- sequence-level API --------------------------------------------------
    def can_allocate(self, seq: Sequence, num_new_tokens: int, computed_blocks: list[int] | None = None) -> bool:
        computed_blocks = computed_blocks or []
        available = self.num_free_blocks - sum(1 for b in computed_blocks if self.blocks[b].ref_count == 0)
        return self._num_blocks_needed(seq, num_new_tokens, len(computed_blocks) * self.block_size) <= available

    def _num_blocks_needed(self, seq: Sequence, num_new_tokens: int, num_cached_tokens: int) -> int:
        total = seq.num_computed_tokens + num_cached_tokens + num_new_tokens
        have = len(seq.block_table) + (num_cached_tokens // self.block_size)
        return max(0, self.num_blocks_for_tokens(total, self.block_size) - have)

    def allocate_slots(
        self,
        seq: Sequence,
        num_new_tokens: int,
        computed_blocks: list[int] | None = None,
    ) -> list[int] | None:
        """Make sure `seq` owns blocks for computed + cached + num_new tokens.

        `computed_blocks` (from get_computed_blocks) are appended to the block
        table with a reference taken. Returns the newly allocated block ids or
        None (and changes nothing) if the pool is exhausted.
        """
        computed_blocks = computed_blocks or []
        num_cached = len(computed_blocks) * self.block_size
        need = self._num_blocks_needed(seq, num_new_tokens, num_cached)
        # Cached blocks that are currently free will leave the free queue when touched.
        available = self.num_free_blocks - sum(1 for b in computed_blocks if self.blocks[b].ref_count == 0)
        if need > available:
            return None
        for b in computed_blocks:
            self._touch(b)
        seq.block_table.extend(computed_blocks)
        new_blocks = [self._allocate_block().block_id for _ in range(need)]
        seq.block_table.extend(new_blocks)
        # Copy-on-write: if the block we are about to append into is shared,
        # give this sequence its own copy first.
        if num_new_tokens > 0:
            first_write_token = seq.num_computed_tokens + num_cached
            idx = first_write_token // self.block_size
            if idx < len(seq.block_table) and idx < len(seq.block_table) - len(new_blocks):
                shared = self.blocks[seq.block_table[idx]]
                if shared.ref_count > 1 and first_write_token % self.block_size != 0:
                    if self.num_free_blocks == 0:
                        # roll back
                        for b in new_blocks:
                            self._release(b)
                        del seq.block_table[len(seq.block_table) - len(new_blocks):]
                        for b in computed_blocks:
                            self._release(b)
                        del seq.block_table[len(seq.block_table) - len(computed_blocks):]
                        return None
                    fresh = self._allocate_block()
                    self._pending_cow.append(CopyOnWrite(shared.block_id, fresh.block_id))
                    self._release(shared.block_id)
                    seq.block_table[idx] = fresh.block_id
                    new_blocks.append(fresh.block_id)
        return new_blocks

    def free(self, seq: Sequence) -> None:
        # Release in reverse so the deepest (least shareable) blocks are evicted first.
        for b in reversed(seq.block_table):
            self._release(b)
        seq.block_table = []

    def fork(self, parent: Sequence, child: Sequence) -> None:
        """Child shares all of parent's blocks (e.g. beam / n>1 sampling)."""
        assert not child.block_table
        for b in parent.block_table:
            self._touch(b)
        child.block_table = list(parent.block_table)
        child.num_computed_tokens = parent.num_computed_tokens

    def take_pending_copies(self) -> list[CopyOnWrite]:
        ops, self._pending_cow = self._pending_cow, []
        return ops

    def slot_mapping(self, seq: Sequence, start_token: int, num_tokens: int) -> list[int]:
        """Flat slot index (block_id * block_size + offset) for a token range."""
        bs = self.block_size
        out = []
        for t in range(start_token, start_token + num_tokens):
            out.append(seq.block_table[t // bs] * bs + t % bs)
        return out
