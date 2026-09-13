"""GPU-only: real checkpoint through the paged engine vs HuggingFace.

Run: pytest tests/test_model_gpu.py -q   (downloads TinyLlama, ~2 GB)
"""
import os

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from pagedllm import EngineConfig, SamplingParams  # noqa: E402
from pagedllm.engine import LLMEngine  # noqa: E402

MODEL = os.environ.get("PAGEDLLM_TEST_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
PROMPTS = [
    "The capital of France is",
    "Write a haiku about GPU memory fragmentation:",
    "def fibonacci(n):",
]


@pytest.fixture(scope="module")
def hf():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()
    return tok, model


def hf_greedy(hf, prompt, n):
    tok, model = hf
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    out = model.generate(ids, max_new_tokens=n, do_sample=False, min_new_tokens=n)
    return out[0, ids.shape[1]:].tolist()


@pytest.mark.parametrize("backend", ["triton", "torch"])
def test_prefill_logits_match_hf(hf, backend):
    tok, model = hf
    engine = LLMEngine(MODEL, EngineConfig(attention_backend=backend, num_gpu_blocks=256, dtype="float16"))
    for prompt in PROMPTS:
        ids = tok(prompt, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            want = model(ids).logits[0, -1].float()
        engine.add_request(ids[0].tolist(), SamplingParams(temperature=0.0, max_tokens=1))
        out = engine.scheduler.schedule()
        inputs = engine.runner.build_inputs(out)
        with torch.no_grad():
            got = engine.model(inputs[0], inputs[1], engine.kv_cache, inputs[2], inputs[3])[0]
        engine.scheduler.update(out, {out.scheduled[0].seq.seq_id: int(got.argmax())})
        assert got.argmax() == want.argmax()
        assert (got - want).abs().max() < 0.5, (got - want).abs().max()


def test_triton_and_torch_backends_agree_on_greedy_decode():
    outs = {}
    for backend in ("triton", "torch"):
        engine = LLMEngine(MODEL, EngineConfig(attention_backend=backend, num_gpu_blocks=512, dtype="float16"))
        outs[backend] = [o.token_ids for o in engine.generate(PROMPTS, SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True))]
        del engine; torch.cuda.empty_cache()
    for a, b in zip(outs["triton"], outs["torch"]):
        assert a == b


def test_greedy_decode_matches_hf(hf):
    engine = LLMEngine(MODEL, EngineConfig(attention_backend="auto", num_gpu_blocks=512, dtype="float16"))
    outs = engine.generate(PROMPTS, SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True))
    for prompt, out in zip(PROMPTS, outs):
        want = hf_greedy(hf, prompt, 32)
        # fp16 argmax can flip on near-ties late in the sequence; demand an exact 16-token prefix.
        assert out.token_ids[:16] == want[:16], (out.text, engine.tokenizer.decode(want))


def test_prefix_cache_hits_on_second_request():
    engine = LLMEngine(MODEL, EngineConfig(num_gpu_blocks=512, dtype="float16", block_size=16))
    prefix = "Once upon a time, in a land far away, " * 8
    a = engine.generate([prefix + "there lived a dragon."], SamplingParams(temperature=0.0, max_tokens=8))[0]
    b = engine.generate([prefix + "there lived a knight."], SamplingParams(temperature=0.0, max_tokens=8))[0]
    assert a.num_cached_tokens == 0 and b.num_cached_tokens >= 64
