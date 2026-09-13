"""Continuous-batching scheduler with chunked prefill and recompute preemption.

Unified prefill/decode (vLLM v1 style): every scheduled sequence contributes
`num_new_tokens` query tokens to one batched forward. A decoding sequence
contributes 1, a prefilling one contributes up to the remaining token budget.

Per step:
  1. RUNNING sequences (arrival order) get their next chunk / decode token.
     If the pool has no block for one, the lowest-priority running sequence
     is preempted (KV dropped, put back at the head of WAITING).
  2. WAITING sequences are admitted FIFO while budget and blocks remain,
     after a prefix-cache lookup that skips already-computed blocks.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from pagedllm.block_manager import BlockManager
from pagedllm.config import EngineConfig
from pagedllm.sequence import Sequence, SequenceStatus


@dataclass
class ScheduledSequence:
    seq: Sequence
    num_new_tokens: int
    # True when this step computes the last known token -> logits at the last
    # query position are needed for sampling. Fixed at schedule time.
    needs_sampling: bool = field(init=False)

    def __post_init__(self) -> None:
        self.needs_sampling = self.seq.num_computed_tokens + self.num_new_tokens == self.seq.num_tokens


@dataclass
class SchedulerOutput:
    scheduled: list[ScheduledSequence] = field(default_factory=list)
    preempted: list[Sequence] = field(default_factory=list)
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.scheduled

    @property
    def total_tokens(self) -> int:
        return self.num_prefill_tokens + self.num_decode_tokens


class Scheduler:
    def __init__(self, config: EngineConfig, block_manager: BlockManager, eos_token_id: int | None):
        self.config = config
        self.bm = block_manager
        self.eos_token_id = eos_token_id
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.finished: list[Sequence] = []
        self.num_preemptions = 0

    # ---- queue management -------------------------------------------------
    def add(self, seq: Sequence) -> None:
        if seq.num_prompt_tokens >= self.config.max_model_len:
            raise ValueError(f"prompt has {seq.num_prompt_tokens} tokens, max_model_len={self.config.max_model_len}")
        if not self.config.enable_chunked_prefill and seq.num_prompt_tokens > self.config.max_num_batched_tokens:
            raise ValueError("prompt longer than max_num_batched_tokens; enable chunked prefill")
        if seq.num_prompt_tokens + seq.sampling_params.max_tokens > self.config.max_model_len:
            # Clamp rather than reject; mirrors vLLM's behaviour of truncating output.
            seq.sampling_params.max_tokens = max(1, self.config.max_model_len - seq.num_prompt_tokens)
        seq.status = SequenceStatus.WAITING
        self.waiting.append(seq)

    def abort(self, seq_id: int) -> None:
        for q in (self.waiting, self.running):
            for seq in list(q):
                if seq.seq_id == seq_id:
                    q.remove(seq)
                    self.bm.free(seq)
                    seq.status = SequenceStatus.FINISHED_ABORTED
                    self.finished.append(seq)
                    return

    @property
    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    @property
    def num_unfinished(self) -> int:
        return len(self.waiting) + len(self.running)

    # ---- scheduling ---------------------------------------------------------
    def _preempt(self, seq: Sequence, out: SchedulerOutput) -> None:
        self.running.remove(seq)
        self.bm.free(seq)
        seq.reset_for_recompute()
        self.waiting.appendleft(seq)
        out.preempted.append(seq)
        self.num_preemptions += 1

    def schedule(self) -> SchedulerOutput:
        out = SchedulerOutput()
        budget = self.config.max_num_batched_tokens
        chunked = self.config.enable_chunked_prefill

        # 1. Running sequences.
        for seq in list(self.running):
            if budget == 0:
                break
            if seq in out.preempted:
                continue
            num_new = seq.num_uncomputed_tokens
            assert num_new >= 1
            if num_new > budget:
                if not chunked:
                    continue  # wait for a step with room
                num_new = budget
            # Allocate, preempting from the tail if needed.
            while True:
                new_blocks = self.bm.allocate_slots(seq, num_new)
                if new_blocks is not None:
                    break
                victim = self.running[-1]
                self._preempt(victim, out)
                if victim is seq:
                    break
            if seq.status is not SequenceStatus.RUNNING:
                continue
            budget -= num_new
            out.scheduled.append(ScheduledSequence(seq, num_new))
            if seq.is_prefill_done:
                out.num_decode_tokens += num_new
            else:
                out.num_prefill_tokens += num_new

        # 2. Waiting sequences (FIFO).
        while self.waiting and budget > 0 and len(self.running) < self.config.max_num_seqs:
            seq = self.waiting[0]
            computed_blocks, num_cached = self.bm.get_computed_blocks(seq)
            num_new = seq.num_tokens - num_cached
            assert num_new >= 1
            if num_new > budget:
                if not chunked:
                    break
                num_new = budget
            new_blocks = self.bm.allocate_slots(seq, num_new, computed_blocks)
            if new_blocks is None:
                break  # pool full: keep FIFO order, try next step
            self.waiting.popleft()
            seq.num_computed_tokens = num_cached
            seq.num_cached_tokens = num_cached
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            budget -= num_new
            out.scheduled.append(ScheduledSequence(seq, num_new))
            out.num_prefill_tokens += num_new
        return out

    # ---- post-processing ---------------------------------------------------------
    def update(self, out: SchedulerOutput, sampled: dict[int, int]) -> list[Sequence]:
        """Advance watermarks, append sampled tokens, retire finished sequences.

        `sampled` maps seq_id -> token for every scheduled seq with needs_sampling.
        Returns the sequences that finished this step.
        """
        finished_now: list[Sequence] = []
        for item in out.scheduled:
            seq = item.seq
            before = seq.num_computed_tokens
            seq.num_computed_tokens += item.num_new_tokens
            self.bm.cache_full_blocks(seq, before, seq.num_computed_tokens)
            if not item.needs_sampling:
                continue
            token = sampled[seq.seq_id]
            seq.append_token(token)
            sp = seq.sampling_params
            if (not sp.ignore_eos and token == self.eos_token_id) or token in sp.stop_token_ids:
                seq.status = SequenceStatus.FINISHED_STOPPED
            elif seq.num_generated_tokens >= sp.max_tokens or seq.num_tokens >= self.config.max_model_len:
                seq.status = SequenceStatus.FINISHED_LENGTH
            if seq.is_finished:
                self.running.remove(seq)
                self.bm.free(seq)
                finished_now.append(seq)
        self.finished.extend(finished_now)
        return finished_now
