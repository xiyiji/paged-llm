from dataclasses import dataclass


@dataclass
class EngineConfig:
    """Engine-wide knobs. Field names mirror vLLM so the mapping is obvious."""

    block_size: int = 16
    # Number of KV blocks. None = derive from gpu_memory_utilization at startup.
    num_gpu_blocks: int | None = None
    gpu_memory_utilization: float = 0.9
    max_num_seqs: int = 256
    # Token budget per scheduler step (prefill chunks + decode tokens).
    max_num_batched_tokens: int = 2048
    max_model_len: int = 4096
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    # "auto" | "triton" | "flash" | "torch"
    attention_backend: str = "auto"
    dtype: str = "auto"
    device: str = "cuda"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.block_size & (self.block_size - 1):
            raise ValueError("block_size must be a power of two (Triton kernel uses // and %)")
        if self.max_num_batched_tokens < self.block_size:
            raise ValueError("max_num_batched_tokens must be >= block_size")
