"""Load a tiny HF-format Llama checkpoint (built with transformers' own LlamaForCausalLM)
through pagedllm's loader + engine on CPU and compare greedy output with HF generate.

Covers weight loading, RoPE convention (incl. llama3 scaling), GQA, tied embeddings.
"""
import os

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import LlamaConfig as HFLlamaConfig, LlamaForCausalLM as HFLlama  # noqa: E402

from pagedllm import EngineConfig, SamplingParams  # noqa: E402
from pagedllm.engine import LLMEngine  # noqa: E402


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return transformers.AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    except Exception as e:  # offline
        pytest.skip(f"tokenizer download failed: {e}")


def build_checkpoint(tmp_path, tokenizer, rope_scaling=None, tie=True, kv_heads=2):
    cfg = HFLlamaConfig(vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=kv_heads, max_position_embeddings=512,
                        rope_theta=10000.0, rope_scaling=rope_scaling, tie_word_embeddings=tie, torch_dtype="float32")
    torch.manual_seed(0)
    model = HFLlama(cfg).eval()
    path = tmp_path / "tiny"
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    return str(path), model


PROMPTS = ["Hello world, this is a test of", "def add(a, b):", "The quick brown fox"]


@pytest.mark.parametrize("rope_scaling,tie,kv_heads", [
    (None, True, 2),
    ({"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 128}, False, 4),
])
def test_engine_matches_hf_generate(tmp_path, tokenizer, rope_scaling, tie, kv_heads):
    path, hf_model = build_checkpoint(tmp_path, tokenizer, rope_scaling, tie, kv_heads)
    engine = LLMEngine(path, EngineConfig(device="cpu", attention_backend="torch", num_gpu_blocks=128, block_size=8,
                                          max_num_batched_tokens=16, dtype="float32"))
    outs = engine.generate(PROMPTS, SamplingParams(temperature=0.0, max_tokens=20, ignore_eos=True))
    for prompt, out in zip(PROMPTS, outs):
        ids = tokenizer(prompt, return_tensors="pt").input_ids
        with torch.no_grad():
            want = hf_model.generate(ids, max_new_tokens=20, min_new_tokens=20, do_sample=False)[0, ids.shape[1]:].tolist()
        assert out.token_ids == want, (out.text, tokenizer.decode(want))
    assert engine.block_manager.num_free_blocks == 128
    assert engine.stats.num_preemptions == 0


def test_loader_rejects_non_llama(tmp_path):
    from pagedllm.model.loader import load_hf_config
    os.makedirs(tmp_path / "x")
    (tmp_path / "x" / "config.json").write_text('{"architectures": ["GPT2LMHeadModel"]}')
    with pytest.raises(ValueError):
        load_hf_config(str(tmp_path / "x"))


def test_profiling_forward_exceeding_rope_table_does_not_crash(tmp_path, tokenizer):
    """Regression: startup profiling ran max_num_batched_tokens positions through a RoPE table
    sized by the checkpoint (TinyLlama: 2048) -> out-of-bounds gather on GPU."""
    path, _ = build_checkpoint(tmp_path, tokenizer)  # max_position_embeddings=512
    engine = LLMEngine(path, EngineConfig(device="cpu", attention_backend="torch", num_gpu_blocks=64, block_size=8,
                                          max_num_batched_tokens=1024, max_model_len=256, dtype="float32"))
    engine._dummy_forward(1024)
