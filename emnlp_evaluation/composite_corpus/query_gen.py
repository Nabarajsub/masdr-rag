"""
Generate a labeled query set for the composite corpus.

Strategy: sample N chunks per source_type, prompt Qwen2.5-7B to write ONE
question whose answer is contained in that chunk. Save:

    queries.json: [
      {
        "query_id": "compq_0001",
        "query": "...",
        "gold_source_type": "github",
        "gold_doc_id": "...",
        "gold_chunk_id": "...",
        "reference_answer": "...",
        "category": "github",
        "query_type": "single_domain"
      },
      ...
    ]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO.parent))

from emnlp_evaluation.llm_providers import get_provider


_PROMPT = """Read this document chunk from an enterprise source ({source_type}). Write ONE specific question that a coworker might ask whose answer is contained in this chunk. Also write the short reference answer.

Chunk:
\"\"\"
{chunk}
\"\"\"

Reply with strict JSON only:
{{"question": "...", "answer": "..."}}"""


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract(text: str) -> Dict[str, str]:
    m = _JSON_RE.search(text)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        try:
            return json.loads(m.group(0).replace("'", '"'))
        except Exception:
            return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(Path(__file__).resolve().parent / "data" / "corpus.parquet"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "data" / "queries.json"))
    ap.add_argument("--per-source", type=int, default=10,
                    help="Number of queries to generate per source_type.")
    ap.add_argument("--llm", default="qwen")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-chars", type=int, default=200,
                    help="Skip chunks shorter than this (low information).")
    args = ap.parse_args()

    random.seed(args.seed)
    df = pd.read_parquet(args.corpus)
    df = df[df["text"].str.len() >= args.min_chars].reset_index(drop=True)

    by_src: Dict[str, pd.DataFrame] = {
        src: g.sample(min(args.per_source * 3, len(g)), random_state=args.seed)
        for src, g in df.groupby("source_type")
    }

    print(f"[querygen] loading {args.llm}...", flush=True)
    llm = get_provider(args.llm)

    out: List[Dict] = []
    qid_counter = 0
    t0 = time.time()
    for src, sample in by_src.items():
        kept = 0
        for _, row in sample.iterrows():
            if kept >= args.per_source:
                break
            chunk_text = row["text"][:1800]
            prompt = _PROMPT.format(source_type=src, chunk=chunk_text)
            res = llm.generate(
                [{"role": "user", "content": prompt}],
                max_new_tokens=200, temperature=0.7,
            )
            obj = _extract(res.text)
            q = (obj.get("question") or "").strip()
            a = (obj.get("answer") or "").strip()
            if not q or not a or len(q) < 8:
                continue
            qid_counter += 1
            out.append({
                "query_id": f"compq_{qid_counter:04d}",
                "query": q,
                "gold_source_type": src,
                "gold_doc_id": row["doc_id"],
                "gold_chunk_id": row["id"],
                "reference_answer": a,
                "category": src,
                "query_type": "single_domain",
            })
            kept += 1
        print(f"[querygen] {src}: {kept} queries (elapsed {time.time()-t0:.0f}s)", flush=True)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[querygen] wrote {args.out} :: {len(out)} queries across {len(by_src)} sources")


if __name__ == "__main__":
    main()
