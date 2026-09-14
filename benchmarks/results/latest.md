
### input 256 / output 128 tokens, shared prefix 0%, NVIDIA GeForce RTX 4090

| backend | batch | output tok/s | torch peak (GiB) | KV capacity (tokens) | notes |
|---|---:|---:|---:|---:|---|
| hf | 1 | 146 | 2.08 | - |  |
| pagedllm | 1 | 135 | 19.26 | 818,608 | preempt=0 prefix_hits=0 kv_blocks=51163 attn=paged_attention_triton |
| vllm | 1 | 357 | - | - |  |
| hf | 8 | 1026 | 2.20 | - |  |
| pagedllm | 8 | 1015 | 19.34 | 818,608 | preempt=0 prefix_hits=0 kv_blocks=51163 attn=paged_attention_triton |
| vllm | 8 | 2238 | - | - |  |
| hf | 32 | 3370 | 2.62 | - |  |
| pagedllm | 32 | 3467 | 19.43 | 818,608 | preempt=0 prefix_hits=0 kv_blocks=51163 attn=paged_attention_triton |
| vllm | 32 | 6128 | - | - |  |
| hf | 64 | 5388 | 3.17 | - |  |
| pagedllm | 64 | 5827 | 19.43 | 818,608 | preempt=0 prefix_hits=0 kv_blocks=51163 attn=paged_attention_triton |
| vllm | 64 | 8126 | - | - |  |

### ragged: input 64-256 / output 32-128 tokens, shared prefix 0%, NVIDIA GeForce RTX 4090

| backend | batch | output tok/s | torch peak (GiB) | KV capacity (tokens) | notes |
|---|---:|---:|---:|---:|---|
| hf | 32 | 1835 | 2.61 | - | padding waste 53% |
| pagedllm | 32 | 2691 | 19.43 | 818,608 | preempt=0 prefix_hits=0 kv_blocks=51163 attn=paged_attention_triton |
| vllm | 32 | 4556 | - | - |  |
| hf | 64 | 2096 | 3.15 | - | padding waste 55% |
| pagedllm | 64 | 4210 | 19.43 | 818,608 | preempt=0 prefix_hits=0 kv_blocks=51163 attn=paged_attention_triton |
| vllm | 64 | 6063 | - | - |  |

### input 256 / output 128 tokens, shared prefix 75%, NVIDIA GeForce RTX 4090

| backend | batch | output tok/s | torch peak (GiB) | KV capacity (tokens) | notes |
|---|---:|---:|---:|---:|---|
| pagedllm | 32 | 3744 | 19.43 | 818,608 | preempt=0 prefix_hits=48 kv_blocks=51163 attn=paged_attention_triton |
| vllm | 32 | 7292 | - | - |  |

Output tok/s counts only each request's target tokens. torch peak = torch.cuda.max_memory_allocated;
paged engines pre-allocate the KV pool up to gpu_memory_utilization, so their peak reflects the pool,
not the minimum needed. KV capacity = blocks x block_size the pool could hold.
