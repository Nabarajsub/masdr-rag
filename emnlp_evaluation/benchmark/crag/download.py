"""
Download the secondary benchmark.

The original plan called for Meta's CRAG (KDD Cup 2024). CRAG is gated on
Hugging Face (Meta-Comprehensive-RAG-Benchmark requires per-user access) and
the raw GitHub release stores its corpora behind git-LFS, so a fully
unauthenticated download is not possible.

We substitute **HotpotQA distractor** as the second public benchmark. It is
also multi-document-per-query (10 paragraphs from different Wikipedia
articles, of which 2 are gold supporting), has well-known leaderboards, and
the dataset is freely loadable from Hugging Face. The paper's "Cross-Domain
Generalization" section uses these results to show the dilution effect is
not specific to WYDOT.

Run from a login node (it has internet; compute nodes do not). The output
file path is still `data/crag.jsonl` so downstream ingest/runner code does
not need renaming.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation",
                    help="HotpotQA split (train / validation).")
    ap.add_argument("--max-rows", type=int, default=2000)
    args = ap.parse_args()

    from datasets import load_dataset

    print(f"[hotpot] loading hotpot_qa distractor split={args.split} (streaming)", flush=True)
    ds = load_dataset("hotpot_qa", "distractor", split=args.split, streaming=True)

    out = DATA_DIR / "crag.jsonl"
    n = 0
    t0 = time.time()
    with open(out, "w") as f:
        for ex in ds:
            # Normalize shape so ingest.py can consume it uniformly.
            ctx = ex.get("context") or {}
            titles = ctx.get("title") or []
            sentences = ctx.get("sentences") or []
            paragraphs = []
            for t, sents in zip(titles, sentences):
                paragraphs.append({
                    "title": t,
                    "page_content": " ".join(sents) if isinstance(sents, list) else str(sents),
                })
            sf = ex.get("supporting_facts") or {}
            gold_titles = sorted(set(sf.get("title") or []))
            row = {
                "query": ex.get("question", ""),
                "answer": ex.get("answer", ""),
                "domain": ex.get("type") or "general",   # bridge / comparison
                "question_type": ex.get("level"),
                "static_or_dynamic": "static",
                "search_results": paragraphs,
                "gold_titles": gold_titles,
            }
            f.write(json.dumps(row) + "\n")
            n += 1
            if args.max_rows and n >= args.max_rows:
                break

    print(f"[hotpot] wrote {out} :: {n} rows in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
