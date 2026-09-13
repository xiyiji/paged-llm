"""Minimal usage.  python examples/generate.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0"""
import argparse
import logging

from pagedllm import EngineConfig, SamplingParams
from pagedllm.engine import LLMEngine

logging.basicConfig(level=logging.INFO, format="%(message)s")

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
ap.add_argument("--backend", default="auto")
ap.add_argument("--max-tokens", type=int, default=64)
ap.add_argument("--temperature", type=float, default=0.0)
args = ap.parse_args()

engine = LLMEngine(args.model, EngineConfig(attention_backend=args.backend))
prompts = [
    "The three laws of robotics are",
    "Explain why paged attention reduces GPU memory fragmentation in one paragraph.",
    "Explain why paged attention reduces GPU memory fragmentation in two sentences.",  # prefix-cache hit on the shared prefix
]
for out in engine.generate(prompts, SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)):
    print(f"--- [{out.finish_reason}] cached_prefix_tokens={out.num_cached_tokens}\n{out.text}\n")
s = engine.stats
print(f"steps={s.num_steps} prefill_tokens={s.num_prefill_tokens} decode_tokens={s.num_decode_tokens} "
      f"preemptions={s.num_preemptions} attention={engine.attention_backend}")
