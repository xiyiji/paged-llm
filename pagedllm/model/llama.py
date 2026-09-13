"""Llama-family decoder (TinyLlama, Llama-2/3/3.x) that writes K/V into the
paged cache and attends through the block table.

Module names follow HuggingFace's so safetensors checkpoints load directly.
Batches are token-major: hidden states are [total_tokens, hidden] with no
padding; sequence boundaries live in AttentionMetadata.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from pagedllm.kv_cache import KVCache


@dataclass
class LlamaConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: dict | None
    max_position_embeddings: int
    tie_word_embeddings: bool
    torch_dtype: str

    @classmethod
    def from_hf(cls, cfg: dict) -> "LlamaConfig":
        heads = cfg["num_attention_heads"]
        # transformers >= 4.57 writes {"rope_parameters": {"rope_theta": ..., "rope_type": ..., ...}};
        # older checkpoints have top-level "rope_theta" plus optional "rope_scaling".
        rope = dict(cfg.get("rope_parameters") or cfg.get("rope_scaling") or {})
        rope_theta = rope.pop("rope_theta", None) or cfg.get("rope_theta", 10000.0)
        rope_type = rope.get("rope_type", rope.get("type", "default"))
        rope_scaling = rope if rope_type != "default" else None
        return cls(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            num_attention_heads=heads,
            num_key_value_heads=cfg.get("num_key_value_heads", heads),
            head_dim=cfg.get("head_dim") or cfg["hidden_size"] // heads,
            rms_norm_eps=cfg.get("rms_norm_eps", 1e-5),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=cfg.get("max_position_embeddings", 4096),
            tie_word_embeddings=cfg.get("tie_word_embeddings", False),
            torch_dtype=cfg.get("torch_dtype", "float16"),
        )


@dataclass
class AttentionMetadata:
    """Everything attention needs besides Q/K/V, built once per step by the runner."""

    slot_mapping: torch.Tensor   # [total_tokens] int64: where each new token's K/V goes
    block_tables: torch.Tensor   # [num_seqs, max_blocks] int32
    cu_seqlens_q: torch.Tensor   # [num_seqs + 1] int32
    seq_lens_k: torch.Tensor     # [num_seqs] int32
    max_q_len: int


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(x.dtype)


def _llama3_inv_freq(inv_freq: torch.Tensor, scaling: dict) -> torch.Tensor:
    factor = scaling["factor"]
    low = scaling.get("low_freq_factor", 1.0)
    high = scaling.get("high_freq_factor", 4.0)
    old_len = scaling.get("original_max_position_embeddings", 8192)
    wavelen = 2 * math.pi / inv_freq
    low_wl, high_wl = old_len / low, old_len / high
    smooth = (old_len / wavelen - low) / (high - low)
    scaled = torch.where(wavelen > low_wl, inv_freq / factor, inv_freq)
    mid = (wavelen <= low_wl) & (wavelen >= high_wl)
    smoothed = (1 - smooth) * inv_freq / factor + smooth * inv_freq
    return torch.where(mid, smoothed, scaled)


class RotaryEmbedding(nn.Module):
    def __init__(self, cfg: LlamaConfig, device):
        super().__init__()
        d = cfg.head_dim
        inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, d, 2, dtype=torch.float32, device=device) / d))
        if cfg.rope_scaling:
            rope_type = cfg.rope_scaling.get("rope_type", cfg.rope_scaling.get("type"))
            if rope_type == "llama3":
                inv_freq = _llama3_inv_freq(inv_freq, cfg.rope_scaling)
            else:
                raise NotImplementedError(f"rope_type={rope_type!r} not supported (default and llama3 only)")
        t = torch.arange(cfg.max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)                      # [P, d/2]
        emb = torch.cat([freqs, freqs], dim=-1)               # HF rotate_half convention
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        h = x.shape[-1] // 2
        return torch.cat([-x[..., h:], x[..., :h]], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor):
        cos = self.cos[positions].unsqueeze(1)                # [T, 1, d]
        sin = self.sin[positions].unsqueeze(1)
        qf, kf = q.float(), k.float()
        q_out = qf * cos + self._rotate_half(qf) * sin
        k_out = kf * cos + self._rotate_half(kf) * sin
        return q_out.to(q.dtype), k_out.to(k.dtype)


class Attention(nn.Module):
    def __init__(self, cfg: LlamaConfig, layer_idx: int, attn_fn):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.scale = self.head_dim ** -0.5
        self.attn_fn = attn_fn
        h, d = cfg.hidden_size, cfg.head_dim
        self.q_proj = nn.Linear(h, self.num_heads * d, bias=False)
        self.k_proj = nn.Linear(h, self.num_kv_heads * d, bias=False)
        self.v_proj = nn.Linear(h, self.num_kv_heads * d, bias=False)
        self.o_proj = nn.Linear(self.num_heads * d, h, bias=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor, rope: RotaryEmbedding,
                kv_cache: KVCache, meta: AttentionMetadata) -> torch.Tensor:
        T = x.shape[0]
        q = self.q_proj(x).view(T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(T, self.num_kv_heads, self.head_dim)
        q, k = rope(q, k, positions)
        # 1. Append the new tokens' K/V to their slots in the paged cache.
        kv_cache.write(self.layer_idx, k, v, meta.slot_mapping)
        # 2. Attend over the whole context through the block table.
        o = self.attn_fn(
            q, kv_cache.k_caches[self.layer_idx], kv_cache.v_caches[self.layer_idx],
            meta.block_tables, meta.cu_seqlens_q, meta.seq_lens_k, meta.max_q_len, self.scale,
        )
        return self.o_proj(o.reshape(T, -1))


class MLP(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: LlamaConfig, layer_idx: int, attn_fn):
        super().__init__()
        self.self_attn = Attention(cfg, layer_idx, attn_fn)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, positions, rope, kv_cache, meta):
        x = x + self.self_attn(self.input_layernorm(x), positions, rope, kv_cache, meta)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class LlamaModel(nn.Module):
    def __init__(self, cfg: LlamaConfig, attn_fn, device):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(cfg, i, attn_fn) for i in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.rotary = RotaryEmbedding(cfg, device)

    def forward(self, input_ids, positions, kv_cache, meta):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, positions, self.rotary, kv_cache, meta)
        return self.norm(x)


class LlamaForCausalLM(nn.Module):
    def __init__(self, cfg: LlamaConfig, attn_fn, device):
        super().__init__()
        self.config = cfg
        self.model = LlamaModel(cfg, attn_fn, device)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, kv_cache: KVCache,
                meta: AttentionMetadata, logits_indices: torch.Tensor) -> torch.Tensor:
        """Returns logits only for the rows in `logits_indices` (last token of each sampled seq)."""
        hidden = self.model(input_ids, positions, kv_cache, meta)
        return self.lm_head(hidden[logits_indices]).float()
