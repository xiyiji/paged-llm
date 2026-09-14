"""LLMEngine: tokenizer + scheduler + model runner + sampler in one step loop."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import torch

from pagedllm.attention import select_backend
from pagedllm.block_manager import BlockManager
from pagedllm.config import EngineConfig
from pagedllm.kv_cache import KVCache
from pagedllm.model.llama import AttentionMetadata, LlamaForCausalLM
from pagedllm.model.loader import load_hf_config, load_model, resolve_dtype, resolve_model_path
from pagedllm.sampler import sample
from pagedllm.scheduler import Scheduler, SchedulerOutput
from pagedllm.sequence import SamplingParams, Sequence

log = logging.getLogger("pagedllm")


@dataclass
class StepOutput:
    seq_id: int
    new_token: int | None
    finished: bool


@dataclass
class RequestOutput:
    seq_id: int
    prompt_token_ids: list[int]
    token_ids: list[int]
    text: str
    finish_reason: str
    num_cached_tokens: int
    num_preemptions: int


@dataclass
class EngineStats:
    num_steps: int = 0
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0
    num_preemptions: int = 0
    step_times: list[float] = field(default_factory=list)


class ModelRunner:
    """Turns a SchedulerOutput into tensors, runs the model, samples."""

    def __init__(self, model: LlamaForCausalLM, kv_cache: KVCache, block_manager: BlockManager,
                 device: torch.device, seed: int = 0):
        self.model = model
        self.kv_cache = kv_cache
        self.bm = block_manager
        self.device = device
        self.generator = torch.Generator(device=device).manual_seed(seed)

    def build_inputs(self, out: SchedulerOutput):
        input_ids, positions, slot_mapping, seq_lens, logits_idx, params = [], [], [], [], [], []
        cu = [0]
        block_tables = []
        for item in out.scheduled:
            seq, n = item.seq, item.num_new_tokens
            start = seq.num_computed_tokens
            input_ids.extend(seq.token_ids[start:start + n])
            positions.extend(range(start, start + n))
            slot_mapping.extend(self.bm.slot_mapping(seq, start, n))
            cu.append(cu[-1] + n)
            seq_lens.append(start + n)
            block_tables.append(seq.block_table)
            if item.needs_sampling:
                logits_idx.append(cu[-1] - 1)
                params.append(seq.sampling_params)
        max_blocks = max(len(bt) for bt in block_tables)
        bt = torch.zeros(len(block_tables), max_blocks, dtype=torch.int32)
        for i, row in enumerate(block_tables):
            bt[i, :len(row)] = torch.tensor(row, dtype=torch.int32)
        dev = self.device
        t = lambda x, dt: torch.tensor(x, dtype=dt, device=dev)
        meta = AttentionMetadata(
            slot_mapping=t(slot_mapping, torch.int64),
            block_tables=bt.to(dev, non_blocking=True),
            cu_seqlens_q=t(cu, torch.int32),
            seq_lens_k=t(seq_lens, torch.int32),
            max_q_len=max(item.num_new_tokens for item in out.scheduled),
        )
        return t(input_ids, torch.int64), t(positions, torch.int64), meta, t(logits_idx, torch.int64), params

    @torch.no_grad()
    def run(self, out: SchedulerOutput) -> dict[int, int]:
        self.kv_cache.apply_copies(self.bm.take_pending_copies())
        input_ids, positions, meta, logits_idx, params = self.build_inputs(out)
        logits = self.model(input_ids, positions, self.kv_cache, meta, logits_idx)
        if not params:
            return {}
        tokens = sample(logits, params, self.generator).tolist()
        sampled_seqs = [item.seq for item in out.scheduled if item.needs_sampling]
        return {seq.seq_id: tok for seq, tok in zip(sampled_seqs, tokens)}


class LLMEngine:
    def __init__(self, model: str, config: EngineConfig | None = None, **overrides):
        self.config = config or EngineConfig(**overrides)
        cfg = self.config
        self.device = torch.device(cfg.device)
        torch.manual_seed(cfg.seed)

        from transformers import AutoTokenizer
        path = resolve_model_path(model)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        hf_cfg = load_hf_config(path)
        self.dtype = resolve_dtype(cfg.dtype, hf_cfg)
        attn_fn = select_backend(cfg.attention_backend)
        self.attention_backend = attn_fn.__name__

        t0 = time.perf_counter()
        self.model = load_model(path, attn_fn, self.device, self.dtype, max_positions=cfg.max_model_len)
        mc = self.model.config
        log.info("loaded %s in %.1fs (dtype=%s, attn=%s)", model, time.perf_counter() - t0, self.dtype, self.attention_backend)

        num_blocks = cfg.num_gpu_blocks or self._profile_num_blocks()
        self.kv_cache = KVCache(mc.num_hidden_layers, num_blocks, cfg.block_size, mc.num_key_value_heads,
                                mc.head_dim, self.dtype, self.device)
        self.block_manager = BlockManager(num_blocks, cfg.block_size, cfg.enable_prefix_caching)
        self.scheduler = Scheduler(cfg, self.block_manager, self.tokenizer.eos_token_id)
        self.runner = ModelRunner(self.model, self.kv_cache, self.block_manager, self.device, cfg.seed)
        self.stats = EngineStats()
        self._seqs: dict[int, Sequence] = {}
        log.info("KV cache: %d blocks x %d tokens = %d tokens (%.2f GiB)", num_blocks, cfg.block_size,
                 num_blocks * cfg.block_size, num_blocks * self._block_bytes() / 2**30)

    # ---- memory profiling ------------------------------------------------------
    def _block_bytes(self) -> int:
        mc = self.model.config
        return KVCache.block_bytes(mc.num_hidden_layers, self.config.block_size, mc.num_key_value_heads, mc.head_dim, self.dtype)

    def _dummy_forward(self, n_tok: int) -> None:
        """One forward of n_tok tokens against a throwaway cache (startup memory profiling)."""
        cfg = self.config
        mc = self.model.config
        tmp_blocks = (n_tok + cfg.block_size - 1) // cfg.block_size
        tmp_cache = KVCache(mc.num_hidden_layers, tmp_blocks, cfg.block_size, mc.num_key_value_heads, mc.head_dim, self.dtype, self.device)
        # Positions must stay inside the RoPE table; the values do not matter for profiling.
        positions = torch.arange(n_tok, device=self.device) % mc.max_position_embeddings
        meta = AttentionMetadata(
            slot_mapping=torch.arange(n_tok, device=self.device),
            block_tables=torch.arange(tmp_blocks, dtype=torch.int32, device=self.device)[None, :],
            cu_seqlens_q=torch.tensor([0, n_tok], dtype=torch.int32, device=self.device),
            seq_lens_k=torch.tensor([n_tok], dtype=torch.int32, device=self.device),
            max_q_len=n_tok,
        )
        with torch.no_grad():
            self.model(torch.zeros(n_tok, dtype=torch.int64, device=self.device), positions, tmp_cache, meta,
                       torch.arange(min(n_tok, cfg.max_num_seqs), device=self.device))
        return tmp_blocks

    def _profile_num_blocks(self) -> int:
        """Like vLLM: run one max-size forward to measure activation peak, give the rest to KV."""
        cfg = self.config
        if self.device.type != "cuda":
            return max(64, cfg.max_model_len * 4 // cfg.block_size)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        tmp_blocks = self._dummy_forward(cfg.max_num_batched_tokens)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated(self.device)
        tmp_bytes = tmp_blocks * self._block_bytes()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(self.device)
        used_by_others = total - free - torch.cuda.memory_allocated(self.device)
        budget = int(total * cfg.gpu_memory_utilization) - used_by_others - (peak - tmp_bytes)
        num_blocks = budget // self._block_bytes()
        if num_blocks < 16:
            raise RuntimeError(f"not enough GPU memory for KV cache ({num_blocks} blocks); lower max_num_batched_tokens or raise gpu_memory_utilization")
        return num_blocks

    # ---- request API -------------------------------------------------------------
    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams | None = None) -> int:
        ids = self.tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
        seq = Sequence(ids, sampling_params or SamplingParams())
        self.scheduler.add(seq)
        self._seqs[seq.seq_id] = seq
        return seq.seq_id

    def abort_request(self, seq_id: int) -> None:
        self.scheduler.abort(seq_id)

    @property
    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_unfinished

    def step(self) -> list[StepOutput]:
        t0 = time.perf_counter()
        out = self.scheduler.schedule()
        if out.is_empty:
            return []
        sampled = self.runner.run(out)
        finished = self.scheduler.update(out, sampled)
        finished_ids = {s.seq_id for s in finished}
        st = self.stats
        st.num_steps += 1
        st.num_prefill_tokens += out.num_prefill_tokens
        st.num_decode_tokens += out.num_decode_tokens
        st.num_preemptions += len(out.preempted)
        st.step_times.append(time.perf_counter() - t0)
        return [StepOutput(item.seq.seq_id, sampled.get(item.seq.seq_id), item.seq.seq_id in finished_ids)
                for item in out.scheduled]

    def _to_output(self, seq: Sequence) -> RequestOutput:
        reason = {"FINISHED_STOPPED": "stop", "FINISHED_LENGTH": "length", "FINISHED_ABORTED": "abort"}[seq.status.name]
        return RequestOutput(seq.seq_id, seq.prompt_token_ids, seq.output_token_ids,
                             self.tokenizer.decode(seq.output_token_ids, skip_special_tokens=True),
                             reason, seq.num_cached_tokens, seq.num_preemptions)

    def generate(self, prompts: list[str] | list[list[int]], sampling_params: SamplingParams | list[SamplingParams] | None = None,
                 use_tqdm: bool = False) -> list[RequestOutput]:
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params or SamplingParams()] * len(prompts)
        ids = [self.add_request(p, sp) for p, sp in zip(prompts, sampling_params)]
        pbar = None
        if use_tqdm:
            from tqdm import tqdm
            pbar = tqdm(total=len(ids), desc="requests")
        while self.has_unfinished_requests:
            for o in self.step():
                if o.finished and pbar:
                    pbar.update(1)
        if pbar:
            pbar.close()
        return [self._to_output(self._seqs.pop(i)) for i in ids]
