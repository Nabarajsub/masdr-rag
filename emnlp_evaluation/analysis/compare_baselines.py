#!/usr/bin/env python3
"""Aggregate judged JSONL files into a per-system comparison table.

For each system, we compute mean correctness, mean faithfulness, mean
llm_calls, and mean wall_time on the intersection of query_ids across
all systems present, so the comparison is apples-to-apples.

Usage:
    python -m emnlp_evaluation.analysis.compare_baselines \
        --in 'emnlp_evaluation/results/composite_qwen_monolithic*judged.jsonl' \
        --in 'emnlp_evaluation/results/composite_qwen_baselines_final.judged.jsonl' \
        --label composite
"""
from __future__ import annotations
import argparse, glob, json
from collections import defaultdict
from pathlib import Path

ORDER = [
    "monolithic", "naive", "regex_scoped", "hybrid_routed", "r2_routed",
    "llm_scoped", "masdr_rag", "masdr_singlecall", "react", "langchain_react",
    "ma_rag", "scout_rag",
    "bm25_only", "bm25_qwen",
    "colbert_only", "colbert_qwen", "colbert_scoped_qwen",
]


def load(patterns):
    files = []
    for p in patterns:
        files.extend(Path(f) for f in glob.glob(p))
    if not files:
        raise SystemExit(f"No files matched: {patterns}")
    print(f"[compare] loading {len(files)} file(s):")
    for f in sorted(files):
        print(f"    {f}")
    recs = []
    for f in files:
        with open(f) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "correctness" not in r:
                    continue
                recs.append(r)
    return recs


def aggregate(recs):
    by_sys = defaultdict(dict)
    for r in recs:
        s = r.get("system", "?")
        q = r.get("query_id")
        if q is None:
            continue
        by_sys[s].setdefault(q, r)
    if not by_sys:
        return {}, set()
    shared = set.intersection(*[set(d.keys()) for d in by_sys.values()])
    out = {}
    for s, by_q in by_sys.items():
        sub = [by_q[q] for q in shared]
        n = len(sub)
        if n == 0:
            continue
        def mean(field):
            vs = [float(r.get(field) or 0) for r in sub]
            return sum(vs) / len(vs) if vs else 0.0
        row = {}
        row["n_shared"] = n
        row["n_total"] = len(by_q)
        row["correctness_pct"] = 100.0 * mean("correctness")
        row["faithfulness"] = mean("faithfulness")
        row["llm_calls"] = mean("llm_calls")
        row["wall_s"] = mean("wall_time_s")
        out[s] = row
    return out, shared


def order_key(s):
    try:
        return ORDER.index(s)
    except ValueError:
        return len(ORDER)


def print_table(label, agg, shared):
    if not agg:
        print(f"\n[{label}] no judged records found")
        return
    print(f"\n[{label}] n_shared = {len(shared)} queries (intersection of all systems)")
    hdr = ["System", "n", "Corr%", "Faith", "Calls", "Wall(s)"]
    widths = [22, 5, 7, 7, 7, 8]
    line = " ".join(h.ljust(w) for h, w in zip(hdr, widths))
    print(line)
    print("-" * len(line))
    for s in sorted(agg, key=order_key):
        r = agg[s]
        cells = [
            s.ljust(widths[0]),
            str(r["n_shared"]).rjust(widths[1]),
            f"{r['correctness_pct']:.1f}".rjust(widths[2]),
            f"{r['faithfulness']:.3f}".rjust(widths[3]),
            f"{r['llm_calls']:.2f}".rjust(widths[4]),
            f"{r['wall_s']:.1f}".rjust(widths[5]),
        ]
        print(" ".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="patterns", action="append", required=True,
                    help="Glob pattern; may be repeated.")
    ap.add_argument("--label", default="comparison")
    args = ap.parse_args()
    recs = load(args.patterns)
    agg, shared = aggregate(recs)
    print_table(args.label, agg, shared)


if __name__ == "__main__":
    main()
