"""S-6: k/chunk/rerank sweep.

For each completed run we already have `chunk_ids` (top-k retrieved chunks
for k=15 by default). We can synthesize results for any k <= 15 by
truncating chunk_ids, recomputing R@k from gold-chunk-id (composite/CRAG)
or relevant_title (WYDOT), and re-evaluating answer correctness at that
truncation. Because we cannot easily re-run synthesis offline without a
GPU, we report retrieval metrics here (R@3, R@5, R@10, R@15) and leave
the synthesis half to a follow-up SLURM run if needed.

Outputs:
    judge_assets/k_sweep.json
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS = Path("<DATA_ROOT>")
OUT = Path("<DATA_ROOT>")
OUT.parent.mkdir(parents=True, exist_ok=True)

KS = (3, 5, 10, 15)


def hit_for_record(r: dict, k: int) -> int | None:
    gold_chunk = r.get("gold_chunk_id")
    if gold_chunk:
        return 1 if gold_chunk in (r.get("chunk_ids") or [])[:k] else 0
    gold_title = r.get("relevant_title")
    if gold_title:
        titles = [(t or "").strip().lower() for t in (r.get("chunk_series") or [])[:k]]
        return 1 if gold_title.strip().lower() in titles else 0
    return None


def main() -> None:
    files = {
        "composite": list(RESULTS.glob("composite_qwen_*.judged.jsonl"))
                   + list(RESULTS.glob("composite_qwen_*.judged.judged.jsonl"))[:0],
        "wydot_bge": list(RESULTS.glob("wydot_qwen_bge_m3_*.judged.jsonl")),
        "wydot_gem": list(RESULTS.glob("wydot_qwen_gemini_*_shard*.judged.jsonl")),
        "crag": list(RESULTS.glob("crag_qwen_*.judged.jsonl")),
    }
    summary = {}
    for corpus, paths in files.items():
        by_sys = defaultdict(lambda: {k: [] for k in KS})
        seen = set()
        for p in paths:
            for line in open(p):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "error" in r:
                    continue
                key = (r.get("query_id"), r.get("system"))
                if key in seen:
                    continue
                seen.add(key)
                sys = r.get("system", "?")
                for k in KS:
                    h = hit_for_record(r, k)
                    if h is not None:
                        by_sys[sys][k].append(h)
        for sys, ks in sorted(by_sys.items()):
            row = {f"R@{k}": (sum(ks[k]) / len(ks[k]) if ks[k] else None)
                   for k in KS}
            row["n_scored"] = len(ks[10]) if ks[10] else 0
            summary[f"{corpus}/{sys}"] = row
            ns = row["n_scored"]
            r3 = row["R@3"] or 0; r5 = row["R@5"] or 0
            r10 = row["R@10"] or 0; r15 = row["R@15"] or 0
            print(f"{corpus:<10} {sys:<22} n={ns:>4}  "
                  f"R@3={r3:.3f}  R@5={r5:.3f}  R@10={r10:.3f}  R@15={r15:.3f}")
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\n[wrote] {OUT}")


if __name__ == "__main__":
    main()
