"""Turn benchmarks/results/*.jsonl into a markdown table.  python -m benchmarks.report results/TinyLlama-1.1B-Chat-v1.0.jsonl"""
import json
import sys
from collections import defaultdict


def main(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    by = defaultdict(dict)
    for r in rows:
        key = (r["input_len"], r["output_len"], r["shared_prefix"], bool(r.get("var_len")))
        by[key][(r["backend"], r["batch_size"])] = r
    for (inp, outp, shared, var), runs in sorted(by.items()):
        shape = f"ragged: input {inp//4}-{inp} / output {outp//4}-{outp}" if var else f"input {inp} / output {outp}"
        print(f"\n### {shape} tokens, shared prefix {shared:.0%}, {rows[0]['gpu']}\n")
        print("| backend | batch | output tok/s | torch peak (GiB) | KV capacity (tokens) | notes |")
        print("|---|---:|---:|---:|---:|---|")
        for (backend, bs), r in sorted(runs.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            if r["status"] != "ok":
                print(f"| {backend} | {bs} | OOM | - | - | {r['status']} |")
                continue
            notes = []
            if "preemptions" in r:
                notes.append(f"preempt={r['preemptions']} prefix_hits={r['prefix_cache_hits']} kv_blocks={r['kv_blocks']} attn={r['attention']}")
            if r.get("vllm_eager"):
                notes.append("eager")
            if "generated_tokens_incl_waste" in r and r["generated_tokens_incl_waste"] != r.get("output_tokens"):
                notes.append(f"padding waste {r['generated_tokens_incl_waste'] / r['output_tokens'] - 1:.0%}")
            tp = f"{r.get('torch_peak_gib', float('nan')):.2f}" if "torch_peak_gib" in r else "-"
            cap = f"{r['kv_blocks'] * r['block_size']:,}" if "kv_blocks" in r else "-"
            print(f"| {backend} | {bs} | {r['output_tok_per_s']:.0f} | {tp} | {cap} | {' '.join(notes)} |")
    print("\nOutput tok/s counts only each request's target tokens. torch peak = torch.cuda.max_memory_allocated;")
    print("paged engines pre-allocate the KV pool up to gpu_memory_utilization, so their peak reflects the pool,")
    print("not the minimum needed. KV capacity = blocks x block_size the pool could hold.")


if __name__ == "__main__":
    main(sys.argv[1])
