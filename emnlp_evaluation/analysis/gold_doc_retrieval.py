"""Gold-DOCUMENT retrieval metrics from judged run logs.

The paper carries two different P@10 definitions and they differ by ~2.6x:

  * scope-purity P@10  -- a chunk is relevant if it belongs to the query's
    target document_series. This is what Tables 3 and 6 report.
  * gold-document P@10 -- a chunk is relevant only if it comes from a document
    that actually contains the answer (`gold_titles`). This is the strict
    metric of App. C's dilution curve, and the one that predicts whether the
    answer-bearing evidence is in the synthesiser's context at all.

This module scores any judged .jsonl against the *second* definition, so every
system is measured on identical criteria. Only queries carrying `gold_titles`
are scorable (32 of the 200-query WYDOT suite).

Usage:
    python -m emnlp_evaluation.analysis.gold_doc_retrieval \
        emnlp_evaluation/results/wydot_hand_qwen_promptctl_*.judged.jsonl
"""
from __future__ import annotations

import argparse, ast, json, sys
from collections import defaultdict


def _gold(row):
    gt = row.get("gold_titles")
    if isinstance(gt, str):
        try:
            gt = ast.literal_eval(gt)
        except (ValueError, SyntaxError):
            gt = [gt]
    return {str(t) for t in gt} if gt else set()


def score(rows, k=10):
    """P@k, R@k (>=1 gold chunk) and MRR of the first gold chunk, per system."""
    per = defaultdict(list)
    for r in rows:
        gold = _gold(r)
        if not gold:
            continue
        titles = [str(t) for t in (r.get("chunk_titles") or [])][:k]
        if not titles:
            per[r.get("system")].append((0.0, 0.0, 0.0, r["query_id"]))
            continue
        hits = [i for i, t in enumerate(titles) if t in gold]
        per[r.get("system")].append((
            len(hits) / k,                        # P@k -- k is the budget, not len(titles)
            1.0 if hits else 0.0,                 # R@k at document level
            1.0 / (hits[0] + 1) if hits else 0.0,  # MRR of first gold chunk
            r["query_id"],
        ))
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()

    rows = []
    for f in a.files:
        rows += [json.loads(l) for l in open(f)]
    per = score(rows, a.k)

    print(f"{'system':<26}{'n':>5}{'P@'+str(a.k):>9}{'R@'+str(a.k):>9}{'MRR':>9}{'zero':>7}")
    for s, v in sorted(per.items(), key=lambda kv: -sum(x[0] for x in kv[1]) / max(1, len(kv[1]))):
        n = len(v)
        p = sum(x[0] for x in v) / n
        r = sum(x[1] for x in v) / n
        m = sum(x[2] for x in v) / n
        z = sum(1 for x in v if x[1] == 0)
        print(f"{s:<26}{n:>5}{p:>9.3f}{r:>9.3f}{m:>9.3f}{z:>7}")
    return per


if __name__ == "__main__":
    main()
