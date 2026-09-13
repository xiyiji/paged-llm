"""Vectorised sampling with per-sequence temperature / top-k / top-p."""

from __future__ import annotations

import torch

from pagedllm.sequence import SamplingParams


@torch.no_grad()
def sample(logits: torch.Tensor, params: list[SamplingParams], generator: torch.Generator | None = None) -> torch.Tensor:
    """logits: [n, vocab] float32. Returns [n] int64 token ids."""
    n, vocab = logits.shape
    assert n == len(params)
    device = logits.device
    temps = torch.tensor([p.temperature for p in params], device=device)
    greedy = temps < 1e-5
    if bool(greedy.all()):
        return logits.argmax(-1)

    logits = logits / temps.clamp(min=1e-5)[:, None]
    top_k = torch.tensor([p.top_k if p.top_k > 0 else vocab for p in params], device=device)
    top_p = torch.tensor([p.top_p for p in params], device=device)
    if bool((top_k < vocab).any()) or bool((top_p < 1.0).any()):
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        ranks = torch.arange(vocab, device=device)[None, :]
        remove = ranks >= top_k[:, None]
        probs = sorted_logits.softmax(-1)
        cum = probs.cumsum(-1) - probs
        remove |= cum > top_p[:, None]
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    probs = logits.softmax(-1)
    sampled = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
    return torch.where(greedy, logits.argmax(-1), sampled)
