from pagedllm.block_manager import BlockManager
from pagedllm.config import EngineConfig
from pagedllm.scheduler import Scheduler
from pagedllm.sequence import SamplingParams, Sequence, SequenceStatus

EOS = 2


def make(num_blocks=64, block_size=4, budget=32, max_seqs=8, chunked=True, prefix=True, max_len=256):
    cfg = EngineConfig(block_size=block_size, max_num_batched_tokens=budget, max_num_seqs=max_seqs,
                       enable_chunked_prefill=chunked, enable_prefix_caching=prefix, max_model_len=max_len,
                       num_gpu_blocks=num_blocks)
    bm = BlockManager(num_blocks, block_size, prefix)
    return cfg, bm, Scheduler(cfg, bm, EOS)


def run_step(sched, next_token=7):
    out = sched.schedule()
    sampled = {s.seq.seq_id: next_token for s in out.scheduled if s.needs_sampling}
    finished = sched.update(out, sampled)
    return out, finished


def test_prefill_then_decode_then_finish():
    _, _, sched = make()
    seq = Sequence(list(range(10, 20)), SamplingParams(max_tokens=3))
    sched.add(seq)
    out, _ = run_step(sched)
    assert out.num_prefill_tokens == 10 and out.num_decode_tokens == 0
    assert seq.status is SequenceStatus.RUNNING and seq.num_generated_tokens == 1
    out, _ = run_step(sched)
    assert out.num_decode_tokens == 1
    out, finished = run_step(sched)
    assert finished == [seq] and seq.status is SequenceStatus.FINISHED_LENGTH
    assert seq.output_token_ids == [7, 7, 7]
    assert not sched.has_unfinished


def test_eos_stops():
    _, bm, sched = make()
    seq = Sequence([5, 6, 7], SamplingParams(max_tokens=10))
    sched.add(seq)
    out = sched.schedule()
    sched.update(out, {seq.seq_id: EOS})
    assert seq.status is SequenceStatus.FINISHED_STOPPED and bm.num_free_blocks == 64


def test_chunked_prefill_respects_budget():
    _, _, sched = make(budget=8)
    seq = Sequence(list(range(100, 120)), SamplingParams(max_tokens=2))
    sched.add(seq)
    out, _ = run_step(sched)
    assert out.scheduled[0].num_new_tokens == 8 and seq.num_computed_tokens == 8
    assert seq.num_generated_tokens == 0  # not sampled mid-prefill
    out, _ = run_step(sched)
    assert seq.num_computed_tokens == 16
    out, _ = run_step(sched)
    assert seq.num_computed_tokens == 20 and seq.num_generated_tokens == 1


def test_mixed_prefill_and_decode_in_one_step():
    _, _, sched = make(budget=16)
    a = Sequence(list(range(4)), SamplingParams(max_tokens=10))
    sched.add(a)
    run_step(sched)                       # a is now decoding
    b = Sequence(list(range(40)), SamplingParams(max_tokens=10))
    sched.add(b)
    out, _ = run_step(sched)
    assert out.num_decode_tokens == 1 and out.num_prefill_tokens == 15
    assert [s.seq for s in out.scheduled] == [a, b]


def test_fifo_admission_and_max_num_seqs():
    _, _, sched = make(max_seqs=2, budget=1000)
    seqs = [Sequence([1, 2, 3], SamplingParams(max_tokens=5)) for _ in range(3)]
    for s in seqs:
        sched.add(s)
    out, _ = run_step(sched)
    assert [x.seq for x in out.scheduled] == seqs[:2]
    assert seqs[2].status is SequenceStatus.WAITING


def test_preemption_when_blocks_run_out_and_recovery():
    # 6 blocks of 4 tokens: two seqs of 8 prompt tokens each fill 4 blocks; decoding
    # past token 8 needs a 3rd block each -> only room for one; the other is preempted.
    _, bm, sched = make(num_blocks=6, block_size=4, budget=1000)
    a = Sequence(list(range(8)), SamplingParams(max_tokens=8))
    b = Sequence(list(range(8, 16)), SamplingParams(max_tokens=8))
    sched.add(a); sched.add(b)
    run_step(sched)                       # both prefilled, both have 1 generated token
    assert bm.num_free_blocks == 2
    run_step(sched)                       # tokens 8 need block 3 each: a gets one, b gets the other
    assert bm.num_free_blocks == 0
    for _ in range(3):                    # tokens 9..11 fit in the third block
        out, _ = run_step(sched)
    assert bm.num_free_blocks == 0 and not out.preempted
    out, _ = run_step(sched)              # token 12 needs a 4th block: b (tail) is preempted
    assert b in out.preempted and b.status is SequenceStatus.WAITING and b.block_table == []
    assert a.status is SequenceStatus.RUNNING
    assert sched.num_preemptions >= 1
    # Run a to completion; b then gets readmitted with its generated tokens intact.
    gen_before = b.num_generated_tokens
    while a.status is SequenceStatus.RUNNING:
        run_step(sched)
    assert a.output_token_ids == [7] * 8
    while sched.has_unfinished:
        run_step(sched)
    assert b.status is SequenceStatus.FINISHED_LENGTH
    assert b.num_generated_tokens == 8 and b.num_preemptions >= 1 and gen_before >= 1
    assert bm.num_free_blocks == 6


def test_prefix_cache_across_requests():
    _, bm, sched = make(block_size=4, budget=1000)
    shared = list(range(50, 66))  # 16 tokens = 4 full blocks
    a = Sequence(shared + [1, 2], SamplingParams(max_tokens=1))
    sched.add(a)
    while sched.has_unfinished:
        run_step(sched)
    b = Sequence(shared + [3, 4, 5], SamplingParams(max_tokens=1))
    sched.add(b)
    out, _ = run_step(sched)
    assert b.num_cached_tokens == 16 and out.scheduled[0].num_new_tokens == 3
    assert bm.num_cache_hits == 1


def test_no_chunking_when_disabled():
    _, _, sched = make(budget=8, chunked=False)
    small = Sequence(list(range(8)), SamplingParams(max_tokens=1))
    sched.add(small)
    out, _ = run_step(sched)
    assert out.scheduled[0].num_new_tokens == 8
