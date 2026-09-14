# pagedllm

A small LLM inference engine whose attention **actually reads K/V through a block table** —
a Triton paged-attention kernel, not a gather-then-dense-attention simulation.

Scope is deliberately narrow and deep: block manager → continuous-batching scheduler →
prefix caching → paged attention kernel, wired into a Llama forward pass that loads
HuggingFace checkpoints. No monitoring, no distributed. ~1.2k lines of engine code,
~700 lines of tests.

```
prompt ──► Scheduler ──► ModelRunner ──► Llama (Triton paged attention) ──► Sampler
              │                              │
         BlockManager ◄──── block tables ────┘        KVCache [num_blocks, block_size, kv_heads, head_dim]
         (ref counts, LRU free list, hash-based prefix cache, copy-on-write)
```

## What is in the box

| Layer | File | What it does |
|---|---|---|
| Block manager | `pagedllm/block_manager.py` | Fixed-size KV blocks, ref-counted; LRU free queue where freed blocks keep their content hash so a later request with the same prefix re-uses them (vLLM v1 style). Copy-on-write when a forked sequence writes into a shared partial block. |
| Scheduler | `pagedllm/scheduler.py` | Continuous batching with a per-step token budget. Prefill and decode share one forward (a decode seq contributes 1 query token, a prefill seq up to the remaining budget → chunked prefill). Preemption by recompute when the pool is exhausted, FIFO re-admission. |
| Kernel | `pagedllm/attention/triton_kernel.py` | One Triton kernel for prefill, chunked prefill and decode. Each program owns a query tile of one sequence/head, walks the context in tiles, resolves `block_table[n // block_size]` per key row, online softmax in fp32. GQA via head → kv_head mapping. |
| Backends | `pagedllm/attention/` | `triton` (ours), `flash` (flash-attn ≥ 2.6 varlen + `block_table`), `torch` (pure-PyTorch oracle that runs on CPU). Same signature, swappable with one config flag. |
| Model | `pagedllm/model/llama.py` | Llama-2/3/3.x/TinyLlama forward, token-major (no padding). HF module names, so safetensors load directly. RoPE incl. Llama-3 scaling. |
| Engine | `pagedllm/engine.py` | Tokenizer + scheduler + runner + sampler; vLLM-like memory profiling to size the KV pool. |

## Quick start (GPU)

```bash
pip install -e ".[gpu,dev]"          # torch + triton
python examples/generate.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

```python
from pagedllm import EngineConfig, SamplingParams
from pagedllm.engine import LLMEngine

engine = LLMEngine("meta-llama/Llama-3.2-1B-Instruct", EngineConfig(attention_backend="triton"))
for out in engine.generate(["Paged attention is"], SamplingParams(temperature=0.0, max_tokens=64)):
    print(out.text, out.num_cached_tokens)
```

## Tests

```bash
pytest -q                     # CPU: block manager, scheduler, reference attention, end-to-end engine
pytest -q tests/test_triton_kernel.py tests/test_model_gpu.py   # GPU
```

The CPU suite runs the *whole* engine (scheduler + block manager + KV cache + attention +
sampling) on a tiny random Llama and checks it reproduces dense recomputation token-for-token,
including under forced preemption and with prefix-cache hits. `test_hf_parity_cpu.py` builds a
tiny checkpoint with transformers' own `LlamaForCausalLM`, saves it, loads it through this
engine and compares greedy output with `model.generate` — so weight loading and the RoPE
convention are verified without a GPU.

The GPU suite checks the Triton kernel against the PyTorch reference over prefill / decode /
mixed batches, GQA shapes, fp16 and bf16, then runs TinyLlama and compares logits and greedy
output with HuggingFace.

## Benchmarks

```bash
bash scripts/runpod_setup.sh    # fresh CUDA pod: installs vllm, flash-attn, runs tests + both benchmarks
python -m benchmarks.bench_kernel                                       # kernel micro-benchmark
python -m benchmarks.bench --backend {hf,pagedllm,vllm} --batch-size 32 # end-to-end, one process per run
bash benchmarks/run_all.sh && cat benchmarks/results/latest.md
```

Workload: 64 random prompts × 256 input tokens, greedy decode of exactly 128 tokens, identical
token ids for all backends; peak memory sampled through NVML so vLLM's preallocation is visible.

### Results (RTX 4090, TinyLlama-1.1B-Chat fp16, 64 requests, greedy, EOS ignored)

Output tokens/s, counting only each request's target tokens. Full tables with memory
columns: [`benchmarks/results/latest.md`](benchmarks/results/latest.md); raw records in the
`.jsonl` next to it.

**Equal lengths** (256 in / 128 out — the best case for static batching):

| batch | HF `generate` | pagedllm | vLLM 0.29 |
|---:|---:|---:|---:|
| 1 | 146 | 135 | 357 |
| 8 | 1026 | 1015 | 2238 |
| 32 | 3370 | 3467 | 6128 |
| 64 | 5388 | 5827 | 8126 |

**Ragged lengths** (input 64–256, output 32–128, random order — where static batching pads to
the longest prompt and keeps decoding until the longest output finishes):

| batch | HF `generate` | pagedllm | vLLM 0.29 |
|---:|---:|---:|---:|
| 32 | 1835 (53% wasted decode steps) | 2691 | 4556 |
| 64 | 2096 (55% wasted decode steps) | 4210 | 6063 |

**75% shared prefix**, batch 32: pagedllm 3744 (48 prefix-cache hits, vs 3467 without sharing),
vLLM 7292.

**Kernel micro-benchmark** (Llama-3-8B attention shape: 32 heads / 8 KV heads / head_dim 128,
fp16, block 16; microseconds): [`benchmarks/results/kernel.md`](benchmarks/results/kernel.md).
Prefill of 2048 tokens: Triton 526 µs vs PyTorch gather-then-attend reference 6568 µs.
Decode batch 8 × 512 context: 50 µs vs 1350 µs. Decode batch 1 × 8192 context: 445 µs vs 931 µs.

How to read this honestly:

* **vs HF generate**: on equal lengths the two are within ±8% — there is nothing for continuous
  batching to win there. On ragged lengths pagedllm is 1.5× (batch 32) to 2.0× (batch 64) faster
  because HF spends over half of its decode steps on rows that are already finished.
* **vs vLLM**: pagedllm reaches 46–72% of vLLM's throughput. The gap has three known causes,
  in order: no CUDA graphs (at batch 1 vLLM is 2.6× faster purely from launch overhead; our step
  is dozens of small Python-launched kernels), a decode kernel that pads each 1-token query to a
  16-row tile and does not split long contexts across SMs (see the kernel table: at batch 1 ×
  8192 the reference is only 2× slower), and no fused RMSNorm / RoPE / SiLU kernels. Note vLLM
  ran from a separate environment (torch 2.13, CUDA graphs on); pagedllm and HF ran on torch 2.8.
* **Memory**: paged engines pre-allocate their KV pool, so "peak memory" is not comparable to
  HF; the meaningful number is capacity (818k tokens on a 4090 for this model) and zero
  preemptions across every run.

Reproduce: `bash scripts/gpu_run.sh` on a CUDA box (needs `VLLM_PYTHON` pointing at an
interpreter with vLLM if it is not in the same environment).

## Design notes worth knowing before an interview

* **Why one kernel for prefill and decode.** vLLM v1 schedules prefill chunks and decode tokens
  into the same forward; the kernel takes `cu_seqlens_q` (query offsets) and `seq_lens_k`
  (context lengths) and derives each query's absolute position as
  `seq_len_k - num_q + i`. Causal masking falls out of that. Cost: decode uses a 16-row query
  tile with 15 padded rows (tl.dot minimum), so a dedicated split-K decode kernel is the obvious
  next optimisation.
* **Why hash-based prefix caching instead of a radix tree.** Full blocks are hashed as
  `hash(prev_block_hash, tokens)`, so lookups are O(blocks) dict probes and eviction is just the
  LRU free queue. SGLang's radix tree gives finer-grained (token-level) sharing at the price of a
  tree walk and its own eviction policy; vLLM chose the hash.
* **Why at least one token is always computed.** A fully-cached prompt still needs logits for
  sampling, so `get_computed_blocks` caps hits at `num_tokens - 1`.
* **Preemption is recompute, not swap.** Swapping to CPU needs a copy engine and complicates the
  scheduler; recompute is what vLLM v1 does by default and is lossless because the generated
  tokens are kept.
* **Memory sizing.** Like vLLM, one max-batch forward is run at startup to measure activation
  peak; everything under `gpu_memory_utilization` that is not weights or activations becomes KV
  blocks.

## Not done (on purpose)

CUDA graphs and a split-K decode kernel (the two items the benchmark says matter most),
swap-based preemption, speculative decoding, quantised KV, tensor parallelism, an
OpenAI-compatible server. Each is a self-contained follow-up.
