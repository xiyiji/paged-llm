import pytest

from pagedllm.block_manager import BlockManager
from pagedllm.sequence import SamplingParams, Sequence


def mk(tokens, **kw):
    return Sequence(list(tokens), SamplingParams(**kw))


def test_allocate_and_free_roundtrip():
    bm = BlockManager(num_blocks=8, block_size=4)
    seq = mk(range(10))
    new = bm.allocate_slots(seq, 10)
    assert new is not None and len(new) == 3
    assert seq.block_table == new
    assert bm.num_free_blocks == 5
    assert bm.slot_mapping(seq, 0, 10) == [b * 4 + o for b in new for o in range(4)][:10]
    bm.free(seq)
    assert bm.num_free_blocks == 8
    assert seq.block_table == []


def test_allocate_returns_none_when_exhausted_and_changes_nothing():
    bm = BlockManager(num_blocks=2, block_size=4)
    a = mk(range(8))
    assert bm.allocate_slots(a, 8) is not None
    b = mk(range(3))
    assert bm.allocate_slots(b, 3) is None
    assert b.block_table == [] and bm.num_free_blocks == 0


def test_decode_append_reuses_partial_block():
    bm = BlockManager(num_blocks=4, block_size=4)
    seq = mk(range(5))
    bm.allocate_slots(seq, 5)
    seq.num_computed_tokens = 5
    assert bm.allocate_slots(seq, 1) == []          # slot 5 fits in block 2
    seq.num_computed_tokens = 6
    seq.num_computed_tokens = 8
    assert len(bm.allocate_slots(seq, 1)) == 1      # token 8 needs a new block


def test_prefix_cache_hit_shares_blocks():
    bm = BlockManager(num_blocks=16, block_size=4)
    a = mk(list(range(10)))
    bm.allocate_slots(a, 10)
    bm.cache_full_blocks(a, 0, 10)   # blocks 0,1 full -> hashed
    assert bm.num_cached_blocks == 2

    b = mk(list(range(8)) + [99, 98, 97])
    hits, n = bm.get_computed_blocks(b)
    assert n == 8 and hits == a.block_table[:2]
    new = bm.allocate_slots(b, b.num_tokens - n, hits)
    assert b.block_table[:2] == a.block_table[:2]
    assert len(new) == 1
    assert bm.blocks[a.block_table[0]].ref_count == 2
    bm.free(a)
    assert bm.blocks[a.block_table[0] if a.block_table else hits[0]].ref_count == 1
    bm.free(b)
    assert bm.num_free_blocks == 16
    # Still cached while free.
    c = mk(list(range(9)))
    hits, n = bm.get_computed_blocks(c)
    assert n == 8


def test_prefix_cache_never_returns_whole_prompt():
    bm = BlockManager(num_blocks=8, block_size=4)
    a = mk(range(8))
    bm.allocate_slots(a, 8)
    bm.cache_full_blocks(a, 0, 8)
    b = mk(range(8))
    hits, n = bm.get_computed_blocks(b)
    assert n == 4  # last block left uncomputed so we still get logits


def test_cached_free_blocks_counted_when_checking_capacity():
    bm = BlockManager(num_blocks=3, block_size=4)
    a = mk(range(8))
    bm.allocate_slots(a, 8)
    bm.cache_full_blocks(a, 0, 8)
    bm.free(a)                      # 3 free, 2 of them cached
    b = mk(list(range(8)) + [1, 2, 3, 4, 5])  # needs 2 cached + 2 new = 4 > 3
    hits, n = bm.get_computed_blocks(b)
    assert n == 8
    assert not bm.can_allocate(b, b.num_tokens - n, hits)
    assert bm.allocate_slots(b, b.num_tokens - n, hits) is None
    assert bm.num_free_blocks == 3


def test_lru_eviction_drops_oldest_cached_block():
    bm = BlockManager(num_blocks=3, block_size=4)
    a = mk(range(4)); bm.allocate_slots(a, 4); bm.cache_full_blocks(a, 0, 4); bm.free(a)
    b = mk(range(10, 14)); bm.allocate_slots(b, 4); bm.cache_full_blocks(b, 0, 4); bm.free(b)
    assert bm.num_cached_blocks == 2
    # Allocating 2 fresh blocks: first takes the untouched block, second evicts a's (older).
    c = mk(range(20, 28)); bm.allocate_slots(c, 8)
    assert bm.get_computed_blocks(mk(list(range(4)) + [7]))[1] == 0
    assert bm.get_computed_blocks(mk(list(range(10, 14)) + [7]))[1] == 4


def test_fork_copy_on_write():
    bm = BlockManager(num_blocks=8, block_size=4)
    parent = mk(range(6)); bm.allocate_slots(parent, 6); parent.num_computed_tokens = 6
    child = mk(range(6)); bm.fork(parent, child)
    assert child.block_table == parent.block_table
    assert bm.blocks[parent.block_table[1]].ref_count == 2
    # Child appends into the shared partial block -> gets its own copy.
    new = bm.allocate_slots(child, 1)
    assert child.block_table[0] == parent.block_table[0]
    assert child.block_table[1] != parent.block_table[1]
    ops = bm.take_pending_copies()
    assert len(ops) == 1 and ops[0].src_block == parent.block_table[1] and ops[0].dst_block == child.block_table[1]
    assert bm.blocks[parent.block_table[1]].ref_count == 1
    # Parent now owns its block exclusively: no further COW.
    assert bm.allocate_slots(parent, 1) == []
    assert bm.take_pending_copies() == []


def test_double_free_is_detected():
    bm = BlockManager(num_blocks=2, block_size=4)
    a = mk(range(4)); bm.allocate_slots(a, 4)
    bid = a.block_table[0]
    bm.free(a)
    with pytest.raises(AssertionError):
        bm._release(bid)
