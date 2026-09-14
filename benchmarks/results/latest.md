
### input 256 / output 128 tokens, shared prefix 0%, NVIDIA GeForce RTX 4090

| backend | batch | output tok/s | torch peak (GiB) | KV capacity (tokens) | notes |
|---|---:|---:|---:|---:|---|
| hf | 1 | 146 | 2.08 | - |  |
| pagedllm | 1 | 136 | 19.26 | 818,608 | preempt=0 prefix_hits=0 kv_blocks=51163 attn=paged_attention_triton |
| hf | 8 | 1037 | 2.20 | - |  |
| pagedllm | 8 | 1022 | 19.34 | 818,608 | preempt=0 prefix_hits=0 kv_blocks=51163 attn=paged_attention_triton |
| hf | 32 | 3348 | 2.62 | - |  |
| pagedllm | 32 | 3460 | 19.43 | 818,608 | preempt=0 prefix_hits=0 kv_blocks=51163 attn=paged_attention_triton |
| hf | 64 | 5436 | 3.17 | - |  |
| pagedllm | 64 | 5970 | 19.43 | 818,608 | preempt=0 prefix_hits=0 kv_blocks=51163 attn=paged_attention_triton |

Output tok/s counts only each request's target tokens. torch peak = torch.cuda.max_memory_allocated;
paged engines pre-allocate the KV pool up to gpu_memory_utilization, so their peak reflects the pool,
not the minimum needed. KV capacity = blocks x block_size the pool could hold.
