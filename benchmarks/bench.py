"""Throughput / memory / concurrency benchmark: HF generate vs pagedllm vs vLLM.

One backend per process (so memory numbers are isolated):

    python -m benchmarks.bench --backend pagedllm --batch-size 32 --model TinyLlama/TinyLlama-1.1B-Chat-v1.0

Writes one JSON line per run to benchmarks/results/<model>.jsonl. Use
benchmarks/report.py to turn the file into a markdown table.

Workload: --num-prompts random-token prompts of --input-len tokens, greedy
decode of exactly --output-len tokens (EOS ignored), optional --shared-prefix
fraction so prefix caching has something to hit. Identical token ids for every
backend.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time

import torch


def make_prompts(tokenizer, num_prompts, input_len, output_len, shared_prefix, var_len, seed=0):
    """Returns (prompts, output_lens). With var_len, input lengths are uniform in
    [input_len/4, input_len] and output lengths in [output_len/4, output_len], in random
    order: the ragged workload where static batching pads and waits for the longest row."""
    g = torch.Generator().manual_seed(seed)
    vocab = min(tokenizer.vocab_size, 32000)
    shared_len = int(input_len * shared_prefix)
    shared = torch.randint(100, vocab, (shared_len,), generator=g).tolist()
    prompts, out_lens = [], []
    for _ in range(num_prompts):
        if var_len:
            in_len = int(torch.randint(max(shared_len + 1, input_len // 4), input_len + 1, (1,), generator=g))
            out_len = int(torch.randint(max(1, output_len // 4), output_len + 1, (1,), generator=g))
        else:
            in_len, out_len = input_len, output_len
        rest = torch.randint(100, vocab, (in_len - shared_len,), generator=g).tolist()
        prompts.append(shared + rest)
        out_lens.append(out_len)
    return prompts, out_lens


class GpuMemoryMonitor:
    """Peak *device* memory used (nvidia-smi view), sampled in a thread. Works for vLLM's subprocesses too."""

    def __init__(self, index=0, interval=0.05):
        import pynvml
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(index)
        self.pynvml = pynvml
        self.interval = interval
        self.peak = 0
        self.baseline = self._used()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _used(self):
        return self.pynvml.nvmlDeviceGetMemoryInfo(self.h).used

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self._used())
            time.sleep(self.interval)

    def __enter__(self):
        self._t.start(); return self

    def __exit__(self, *a):
        self._stop.set(); self._t.join()

    @property
    def peak_gib(self):
        return self.peak / 2**30


def run_hf(args, prompts, out_lens, tokenizer):
    """Static batching: left-pad each batch to its longest prompt and decode until its longest
    output is done. Rows that should have stopped earlier keep generating (padding waste);
    only their target tokens count as useful output."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).cuda().eval()
    bs = args.batch_size
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    generated_total = 0
    t0 = time.perf_counter()
    for i in range(0, len(prompts), bs):
        batch = prompts[i:i + bs]
        max_in = max(len(p) for p in batch)
        ids = torch.full((len(batch), max_in), pad_id, dtype=torch.long)
        attn = torch.zeros_like(ids)
        for r, p in enumerate(batch):
            ids[r, max_in - len(p):] = torch.tensor(p)
            attn[r, max_in - len(p):] = 1
        n_out = max(out_lens[i:i + bs])
        with torch.no_grad():
            model.generate(ids.cuda(), attention_mask=attn.cuda(), max_new_tokens=n_out, min_new_tokens=n_out,
                           do_sample=False, pad_token_id=pad_id)
        generated_total += n_out * len(batch)
    torch.cuda.synchronize()
    return time.perf_counter() - t0, {"torch_peak_gib": torch.cuda.max_memory_allocated() / 2**30, "max_concurrency": bs,
                                      "generated_tokens_incl_waste": generated_total}


def run_pagedllm(args, prompts, out_lens, tokenizer):
    from pagedllm import EngineConfig, SamplingParams
    from pagedllm.engine import LLMEngine
    engine = LLMEngine(args.model, EngineConfig(
        attention_backend=args.attention_backend, max_num_seqs=args.batch_size, dtype="float16",
        block_size=args.block_size, gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_batched_tokens=args.max_num_batched_tokens, max_model_len=args.input_len + args.output_len,
        enable_prefix_caching=not args.no_prefix_caching))
    sps = [SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True) for n in out_lens]
    t0 = time.perf_counter()
    outs = engine.generate(prompts, sps)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    assert all(len(o.token_ids) == n for o, n in zip(outs, out_lens))
    capacity = engine.kv_cache.num_blocks * args.block_size
    return elapsed, {
        "torch_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
        "kv_blocks": engine.kv_cache.num_blocks,
        "max_concurrency": min(args.batch_size, capacity // (args.input_len + args.output_len)),
        "num_steps": engine.stats.num_steps,
        "preemptions": engine.stats.num_preemptions,
        "prefix_cache_hits": engine.block_manager.num_cache_hits,
        "attention": engine.attention_backend,
    }


def run_vllm(args, prompts, out_lens, tokenizer):
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, dtype="float16", max_num_seqs=args.batch_size, block_size=args.block_size,
              gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.input_len + args.output_len,
              max_num_batched_tokens=args.max_num_batched_tokens, enable_prefix_caching=not args.no_prefix_caching,
              enforce_eager=args.vllm_eager)
    sps = [SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True) for n in out_lens]
    t0 = time.perf_counter()
    llm.generate([{"prompt_token_ids": p} for p in prompts], sps, use_tqdm=False)
    elapsed = time.perf_counter() - t0
    return elapsed, {"max_concurrency": args.batch_size, "note": "vLLM preallocates KV; see nvidia-smi peak"}


RUNNERS = {"hf": run_hf, "pagedllm": run_pagedllm, "vllm": run_vllm}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=RUNNERS, required=True)
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--num-prompts", type=int, default=64)
    ap.add_argument("--input-len", type=int, default=256)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--shared-prefix", type=float, default=0.0)
    ap.add_argument("--var-len", action="store_true", help="ragged workload: random input/output lengths per request")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-num-batched-tokens", type=int, default=4096)
    ap.add_argument("--attention-backend", default="auto")
    ap.add_argument("--no-prefix-caching", action="store_true")
    ap.add_argument("--vllm-eager", action="store_true", help="disable CUDA graphs in vLLM for an apples-to-apples eager comparison")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts, out_lens = make_prompts(tokenizer, args.num_prompts, args.input_len, args.output_len, args.shared_prefix, args.var_len)
    total_out = sum(out_lens)
    total_in = sum(len(p) for p in prompts)

    record = {k: v for k, v in vars(args).items() if k != "out"}
    record["gpu"] = torch.cuda.get_device_name(0)
    try:
        with GpuMemoryMonitor() as mon:
            elapsed, extra = RUNNERS[args.backend](args, prompts, out_lens, tokenizer)
        record.update(extra)
        record["nvml_peak_gib"] = mon.peak_gib
        record["elapsed_s"] = elapsed
        record["input_tokens"] = total_in
        record["output_tokens"] = total_out
        record["output_tok_per_s"] = total_out / elapsed
        record["total_tok_per_s"] = (total_in + total_out) / elapsed
        record["status"] = "ok"
    except torch.cuda.OutOfMemoryError as e:
        record["status"] = "oom"
        record["error"] = str(e)[:200]
    print(json.dumps(record, indent=2))
    out = args.out or os.path.join(os.path.dirname(__file__), "results", args.model.split("/")[-1] + ".jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "a") as f:
        f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
