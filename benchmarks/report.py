"""Turn benchmarks/results/*.jsonl into a markdown table.  python -m benchmarks.report results/TinyLlama-1.1B-Chat-v1.0.jsonl"""
import json
import sys
from collections import defaultdict


def main(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    by = defaultdict(dict)
    for r in rows:
        key = (r["input_len"], r["output_len"], r["shared_prefix"])
        by[key][(r["backend"], r["batch_size"])] = r
    for (inp, outp, shared), runs in sorted(by.items()):
        print(f"\n### input {inp} / output {outp} tokens, shared prefix {shared:.0%}, {rows[0]['gpu']}\n")
        print("| backend | batch | output tok/s | peak GPU mem (GiB, nvidia-smi) | torch peak (GiB) | notes |")
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
            tp = f"{r.get('torch_peak_gib', float('nan')):.2f}" if "torch_peak_gib" in r else "-"
            print(f"| {backend} | {bs} | {r['output_tok_per_s']:.0f} | {r['nvml_peak_gib']:.2f} | {tp} | {' '.join(notes)} |")


if __name__ == "__main__":
    main(sys.argv[1])
