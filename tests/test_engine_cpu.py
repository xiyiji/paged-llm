"""End-to-end scheduler + block manager + KV cache + attention on CPU with a tiny random Llama.

No checkpoint needed: we build a 2-layer model with random weights and check
that the paged engine reproduces a plain dense forward pass token-for-token.
"""
import math

import pytest
import torch

from pagedllm.attention.reference import paged_attention_reference
from pagedllm.block_manager import BlockManager
from pagedllm.config import EngineConfig
from pagedllm.engine import ModelRunner
from pagedllm.kv_cache import KVCache
from pagedllm.model.llama import AttentionMetadata, LlamaConfig, LlamaForCausalLM
from pagedllm.scheduler import Scheduler
from pagedllm.sequence import SamplingParams, Sequence

CFG = LlamaConfig(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-5,
                  rope_theta=10000.0, rope_scaling=None, max_position_embeddings=256,
                  tie_word_embeddings=True, torch_dtype="float32")


def dense_greedy(model, prompt, max_tokens, block_size=8):
    """Oracle: recompute the full prompt each step with a private single-sequence cache."""
    toks = list(prompt)
    for _ in range(max_tokens):
        n = len(toks)
        nb = (n + block_size - 1) // block_size
        cache = KVCache(CFG.num_hidden_layers, nb, block_size, CFG.num_key_value_heads, CFG.head_dim, torch.float32, "cpu")
        meta = AttentionMetadata(torch.arange(n), torch.arange(nb, dtype=torch.int32)[None], torch.tensor([0, n], dtype=torch.int32),
                                 torch.tensor([n], dtype=torch.int32), n)
        with torch.no_grad():
            logits = model(torch.tensor(toks), torch.arange(n), cache, meta, torch.tensor([n - 1]))
        toks.append(int(logits.argmax(-1)))
    return toks[len(prompt):]


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return LlamaForCausalLM(CFG, paged_attention_reference, "cpu").eval()


def run_engine(model, prompts, max_tokens, block_size=8, budget=24, num_blocks=64, chunked=True, prefix=True, max_seqs=8, sequential=False):
    cfg = EngineConfig(block_size=block_size, max_num_batched_tokens=budget, max_num_seqs=max_seqs, num_gpu_blocks=num_blocks,
                       enable_chunked_prefill=chunked, enable_prefix_caching=prefix, device="cpu", max_model_len=256)
    bm = BlockManager(num_blocks, block_size, prefix)
    cache = KVCache(CFG.num_hidden_layers, num_blocks, block_size, CFG.num_key_value_heads, CFG.head_dim, torch.float32, "cpu")
    sched = Scheduler(cfg, bm, eos_token_id=None)
    runner = ModelRunner(model, cache, bm, torch.device("cpu"))
    seqs = [Sequence(p, SamplingParams(temperature=0.0, max_tokens=max_tokens)) for p in prompts]
    steps = 0
    # sequential=True feeds requests one after another (so later ones can hit the prefix cache).
    for group in ([[s] for s in seqs] if sequential else [seqs]):
        for s in group:
            sched.add(s)
        while sched.has_unfinished:
            out = sched.schedule()
            sched.update(out, runner.run(out))
            steps += 1
    assert bm.num_free_blocks == num_blocks
    return [s.output_token_ids for s in seqs], sched, steps


def test_batched_paged_generation_matches_dense(model):
    g = torch.Generator().manual_seed(1)
    prompts = [torch.randint(0, 97, (n,), generator=g).tolist() for n in (5, 30, 13, 41)]
    got, sched, _ = run_engine(model, prompts, max_tokens=12)
    for p, o in zip(prompts, got):
        assert o == dense_greedy(model, p, 12)


def test_prefix_cache_gives_same_tokens_and_hits(model):
    g = torch.Generator().manual_seed(2)
    shared = torch.randint(0, 97, (24,), generator=g).tolist()
    prompts = [shared + [3, 4], shared + [5, 6, 7, 8]]
    got, sched, _ = run_engine(model, prompts, max_tokens=8, budget=1000, sequential=True)
    assert sched.bm.num_cache_hits == 1 and prompts and sched.finished[1].num_cached_tokens == 24
    for p, o in zip(prompts, got):
        assert o == dense_greedy(model, p, 8)


def test_preemption_under_memory_pressure_is_lossless(model):
    g = torch.Generator().manual_seed(3)
    prompts = [torch.randint(0, 97, (n,), generator=g).tolist() for n in (20, 22, 18, 25)]
    # 14 blocks x 8 = 112 token slots for 4 seqs that grow to ~40 tokens each -> forced preemption.
    got, sched, _ = run_engine(model, prompts, max_tokens=16, num_blocks=14, prefix=False)
    assert sched.num_preemptions > 0
    for p, o in zip(prompts, got):
        assert o == dense_greedy(model, p, 16)


def test_fork_copy_on_write_is_applied(model):
    """Two children forked from a running parent (which owns a partial block) must diverge correctly."""
    block_size, num_blocks = 8, 32
    bm = BlockManager(num_blocks, block_size, enable_prefix_caching=False)
    cache = KVCache(CFG.num_hidden_layers, num_blocks, block_size, CFG.num_key_value_heads, CFG.head_dim, torch.float32, "cpu")
    cfg = EngineConfig(block_size=block_size, num_gpu_blocks=num_blocks, device="cpu", max_model_len=256)
    sched = Scheduler(cfg, bm, None)
    runner = ModelRunner(model, cache, bm, torch.device("cpu"))

    base = list(range(1, 12))  # 11 tokens: block 0 full, block 1 has 3 slots used
    root = Sequence(base, SamplingParams(temperature=0.0, max_tokens=100))
    sched.add(root)
    out = sched.schedule(); sched.update(out, runner.run(out))   # root prefilled, 1 token sampled
    assert len(root.block_table) == 2

    children = []
    for branch_token in (50, 60):
        child = Sequence(base, SamplingParams(temperature=0.0, max_tokens=3))
        bm.fork(root, child)               # shares both blocks incl. the partial one
        child.token_ids = base + [branch_token]
        child.status = root.status
        sched.running.append(child)
        children.append(child)
    sched.abort(root.seq_id)
    assert bm.blocks[children[0].block_table[1]].ref_count == 2

    while sched.has_unfinished:
        out = sched.schedule(); sched.update(out, runner.run(out))
    assert children[0].block_table == [] and children[1].block_table == []
    for child, tok in zip(children, (50, 60)):
        assert child.output_token_ids == dense_greedy(model, base + [tok], 3)
    assert bm.num_free_blocks == num_blocks
