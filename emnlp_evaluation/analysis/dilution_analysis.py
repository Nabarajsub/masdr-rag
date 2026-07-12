"""
Dilution factor per category / source.

Definition (from the CoLM draft):
    dilution(c) = 1 - precision_scoped(c) / precision_monolithic(c)

where precision is the fraction of retrieved chunks whose document_series
(WYDOT) or source (MultiHop-RAG) matches the query category.

Output: a markdown table per benchmark.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path


def _category_field(rec):
    return rec.get("category") or rec.get("query_type") or "ALL"


def _match_field(rec):
    """Per-record: list of strings to match the category against (chunk source/series)."""
    if rec.get("chunk_series"):
        return rec["chunk_series"]
    if rec.get("chunk_sources"):
        return rec["chunk_sources"]
    return []


def _precision(rec):
    cat = _category_field(rec)
    series = _match_field(rec)
    if not series:
        return 0.0
    hits = sum(1 for s in series if s and cat.lower().replace("_", " ") in (s or "").lower())
    return hits / len(series)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="results/*.jsonl")
    args = ap.parse_args()

    base = Path(__file__).resolve().parents[1]
    rows = []
    for p in glob.glob(str(base / args.inputs)):
        with open(p) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # Group: (benchmark, category, system) -> mean precision
    cell = defaultdict(list)
    for r in rows:
        if "error" in r: continue
        cat = _category_field(r)
        bench = r.get("benchmark", "wydot")
        sys_name = r.get("system", "?")
        cell[(bench, cat, sys_name)].append(_precision(r))

    means = {k: sum(v) / len(v) if v else 0 for k, v in cell.items()}

    # Dilution per (benchmark, category): 1 - p_scoped / p_monolithic
    # "scoped" = the best non-monolithic system in that cell.
    by_cat = defaultdict(dict)
    for (bench, cat, sys_name), mean in means.items():
        by_cat[(bench, cat)][sys_name] = mean

    print("| benchmark | category | monolithic | best_scoped | dilution |")
    print("|---|---|---|---|---|")
    for (bench, cat), per_sys in sorted(by_cat.items()):
        mono = per_sys.get("monolithic", float("nan"))
        scoped = max(
            (v for k, v in per_sys.items() if k != "monolithic"),
            default=float("nan"),
        )
        if math.isnan(mono) or math.isnan(scoped) or scoped == 0:
            dil = float("nan")
        else:
            dil = 1 - (mono / scoped)
        dil_str = "nan" if math.isnan(dil) else f"{dil:.3f}"
        print(f"| {bench} | {cat} | {mono:.3f} | {scoped:.3f} | {dil_str} |")


if __name__ == "__main__":
    main()
