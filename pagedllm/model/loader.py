"""Download (if needed) and load a HuggingFace Llama checkpoint into LlamaForCausalLM."""

from __future__ import annotations

import glob
import json
import os

import torch
from safetensors import safe_open

from pagedllm.model.llama import LlamaConfig, LlamaForCausalLM

SUPPORTED_ARCHS = {"LlamaForCausalLM"}


def resolve_model_path(model: str) -> str:
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download
    return snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.model", "*.txt"])


def load_hf_config(path: str) -> dict:
    with open(os.path.join(path, "config.json")) as f:
        cfg = json.load(f)
    archs = cfg.get("architectures", [])
    if not any(a in SUPPORTED_ARCHS for a in archs):
        raise ValueError(f"unsupported architecture {archs}; this engine implements Llama-family models only")
    return cfg


def resolve_dtype(requested: str, cfg: dict) -> torch.dtype:
    if requested == "auto":
        requested = cfg.get("torch_dtype", "float16")
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[requested]


def load_model(path: str, attn_fn, device: torch.device, dtype: torch.dtype) -> LlamaForCausalLM:
    hf_cfg = load_hf_config(path)
    cfg = LlamaConfig.from_hf(hf_cfg)
    with torch.device(device):
        torch.set_default_dtype(dtype)
        try:
            model = LlamaForCausalLM(cfg, attn_fn, device)
        finally:
            torch.set_default_dtype(torch.float32)
    params = dict(model.named_parameters())
    loaded = set()
    for file in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with safe_open(file, framework="pt", device=str(device)) as f:
            for name in f.keys():
                if name not in params:
                    if name.endswith("rotary_emb.inv_freq"):
                        continue
                    raise KeyError(f"unexpected weight {name}")
                with torch.no_grad():
                    params[name].copy_(f.get_tensor(name).to(dtype))
                loaded.add(name)
    missing = set(params) - loaded
    if cfg.tie_word_embeddings:
        missing.discard("lm_head.weight")
    if missing:
        raise KeyError(f"missing weights: {sorted(missing)[:5]} ...")
    model.eval()
    return model
