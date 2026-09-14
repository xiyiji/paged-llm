|  _  / | | | '_ \| '_ \ / _ \ / _` |
|_|  \_\__,_|_| |_| .__/ \___/ \__,_|
GPU: NVIDIA GeForce RTX 4090  heads=32 kv_heads=8 head_dim=128 block_size=16
| workload | batch | context | triton (us) | torch ref (us) |
|---|---:|---:|---:|---:|
| decode | 1 | 512 | 39 | 230 |
| decode | 1 | 2048 | 135 | 229 |
| decode | 1 | 8192 | 445 | 931 |
| decode | 8 | 512 | 50 | 1350 |
| decode | 8 | 2048 | 162 | - |
| decode | 8 | 8192 | 529 | - |
| decode | 32 | 512 | 125 | - |
| decode | 32 | 2048 | 425 | - |
| decode | 32 | 8192 | 1479 | - |
| decode | 64 | 512 | 208 | - |
| decode | 64 | 2048 | 669 | - |
| decode | 64 | 8192 | 2540 | - |
| prefill | 1 | 512 | 66 | 369 |
| prefill | 1 | 2048 | 526 | 6568 |
| chunked prefill (512 new) | 4 | 2048 | 808 | 7103 |
