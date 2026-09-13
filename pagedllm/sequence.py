from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from enum import Enum, auto


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 128
    ignore_eos: bool = False
    stop_token_ids: tuple[int, ...] = ()

    @property
    def greedy(self) -> bool:
        return self.temperature < 1e-5


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED_STOPPED = auto()  # EOS / stop token
    FINISHED_LENGTH = auto()   # max_tokens or max_model_len
    FINISHED_ABORTED = auto()

    @property
    def finished(self) -> bool:
        return self in (
            SequenceStatus.FINISHED_STOPPED,
            SequenceStatus.FINISHED_LENGTH,
            SequenceStatus.FINISHED_ABORTED,
        )


_seq_counter = itertools.count()


@dataclass
class Sequence:
    """One request. `token_ids` = prompt + generated so far.

    `num_computed_tokens` is the number of leading tokens whose KV already
    lives in the cache. Everything between it and len(token_ids) still needs
    a forward pass. For a fresh request this is 0 (or the prefix-cache hit
    length); for a decoding request it is len(token_ids) - 1.
    """

    prompt_token_ids: list[int]
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    seq_id: int = field(default_factory=lambda: next(_seq_counter))
    token_ids: list[int] = field(init=False)
    status: SequenceStatus = SequenceStatus.WAITING
    num_computed_tokens: int = 0
    block_table: list[int] = field(default_factory=list)
    arrival_time: float = field(default_factory=time.perf_counter)
    # Stats
    num_cached_tokens: int = 0   # prefix-cache hits on (last) admission
    num_preemptions: int = 0

    def __post_init__(self) -> None:
        if not self.prompt_token_ids:
            raise ValueError("empty prompt")
        self.token_ids = list(self.prompt_token_ids)

    # ---- sizes -----------------------------------------------------------
    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def num_generated_tokens(self) -> int:
        return len(self.token_ids) - len(self.prompt_token_ids)

    @property
    def num_uncomputed_tokens(self) -> int:
        return self.num_tokens - self.num_computed_tokens

    @property
    def output_token_ids(self) -> list[int]:
        return self.token_ids[len(self.prompt_token_ids):]

    @property
    def is_finished(self) -> bool:
        return self.status.finished

    @property
    def is_prefill_done(self) -> bool:
        return self.num_computed_tokens >= self.num_prompt_tokens

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)

    def reset_for_recompute(self) -> None:
        """Preemption by recompute: drop cached KV, keep generated tokens."""
        self.num_computed_tokens = 0
        self.block_table = []
        self.status = SequenceStatus.WAITING
        self.num_preemptions += 1

    def __hash__(self) -> int:
        return self.seq_id

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Sequence) and other.seq_id == self.seq_id
