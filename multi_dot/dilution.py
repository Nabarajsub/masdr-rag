#!/usr/bin/env python3
"""Phase 3: vector-search dilution analysis (retrieval-free, no LLM, no Neo4j).

Implements the paper's dilution factor directly on the embedding space:

    delta(c) = 1 - P_global(c) / P_scoped(c)

where P_scoped ~= 1 (search restricted to the correct category returns only
in-category chunks) and P_global is measured as **k-NN category purity** on the
monolithic index: for each chunk, the fraction of its top-k nearest neighbours
that share its category. High delta == severe dilution.

For the merged 'alldot' corpus the purity is computed on three axes:
  - document_series  (same category, any DOT)
  - dot              (same DOT, any category)
  - dot x series     (the two-axis scope MASDR-RAG would use)

Inputs : <prefix>.meta.parquet + <prefix>.embeddings.npy  (L2-normalised).
Outputs: <prefix>_dilution.json + a markdown table on stdout.

Usage:
    python -m graph_processing.multi_dot.dilution --corpus cdot
    python -m graph_processing.multi_dot.dilution --corpus alldot --k 10
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

_WS = re.compile(r"\s+")


def _norm_label(v: str) -> str:
    return _WS.sub(" ", str(v)).strip().upper() if v is not None else ""


def _compose(meta: pd.DataFrame, fields):
    """Join 1+ metadata columns with '::' into a single label per row."""
    cols = [meta[f].astype(str).map(_norm_label).to_numpy() for f in fields]
    if len(cols) == 1:
        return cols[0]
    return np.array(["::".join(t) for t in zip(*cols)])

REPO = Path(__file__).resolve().parents[2]


def prefix_for(corpus: str) -> Path:
    sub = "_combined" if corpus == "alldot" else corpus
    return REPO / "data" / sub / "derived" / corpus


def knn_purity(embs: np.ndarray, labels: np.ndarray, k: int,
               batch: int = 512) -> np.ndarray:
    """Per-chunk fraction of top-k neighbours (excl. self) with the same label."""
    n = embs.shape[0]
    purity = np.zeros(n, dtype=np.float32)
    for i in range(0, n, batch):
        sims = embs[i:i + batch] @ embs.T            # cosine (vectors are unit norm)
        rows = np.arange(sims.shape[0])
        sims[rows, np.arange(i, i + sims.shape[0])] = -2.0   # mask self
        nn = np.argpartition(-sims, k, axis=1)[:, :k]
        same = labels[nn] == labels[i:i + sims.shape[0]][:, None]
        purity[i:i + sims.shape[0]] = same.mean(axis=1)
    return purity


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = math.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom else float("nan")


def analyse_axis(embs, labels, k, axis_name):
    purity = knn_purity(embs, labels, k)
    rows, cats, deltas = [], [], []
    for cat in sorted(pd.unique(labels)):
        mask = labels == cat
        p = float(purity[mask].mean())
        delta = 1.0 - p                       # P_scoped ~= 1.0
        rows.append({"category": str(cat), "chunks": int(mask.sum()),
                     "knn_purity": round(p, 4), "dilution_delta": round(delta, 4)})
        cats.append(int(mask.sum()))
        deltas.append(delta)
    overall = float(purity.mean())
    rho = spearman(np.log10(np.maximum(cats, 1)), deltas)
    return {"axis": axis_name, "k": k, "overall_purity": round(overall, 4),
            "overall_dilution": round(1 - overall, 4),
            "spearman_logchunks_delta": round(rho, 4) if not math.isnan(rho) else None,
            "per_category": rows}


def md_table(result) -> str:
    out = [f"\n### Dilution — axis: {result['axis']}  (k={result['k']})",
           f"overall k-NN purity = {result['overall_purity']}  |  "
           f"overall dilution = {result['overall_dilution']}  |  "
           f"Spearman(log chunks, delta) = {result['spearman_logchunks_delta']}",
           "", "| Category | Chunks | k-NN purity | Dilution delta |",
           "|---|---:|---:|---:|"]
    for r in sorted(result["per_category"], key=lambda x: -x["dilution_delta"]):
        out.append(f"| {r['category']} | {r['chunks']} | "
                   f"{r['knn_purity']} | {r['dilution_delta']} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True,
                    help="cdot | caltrans | wydot | alldot")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--scope", default="",
                    help="semicolon-separated list of scope specs; each spec is "
                         "a comma-joined field list composed with '::'. "
                         "Example: 'document_series;document_series,section'. "
                         "If empty, defaults are used per corpus.")
    ap.add_argument("--restrict-to", default="",
                    help="filter chunks by field=value, e.g. "
                         "'document_series=Standard Specs'")
    ap.add_argument("--tag", default="",
                    help="suffix for the output file, e.g. '_section'")
    args = ap.parse_args()

    prefix = prefix_for(args.corpus)
    meta = pd.read_parquet(f"{prefix}.meta.parquet")
    embs = np.load(f"{prefix}.embeddings.npy").astype(np.float32)

    if args.restrict_to:
        field, val = args.restrict_to.split("=", 1)
        keep = meta[field].astype(str) == val
        n_before = len(meta)
        meta = meta[keep].reset_index(drop=True)
        embs = embs[keep.to_numpy()]
        print(f"[{args.corpus}] restricted on {args.restrict_to}: "
              f"{n_before} -> {len(meta)} chunks")

    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.clip(norms, 1e-9, None)
    print(f"[{args.corpus}] {len(meta)} chunks, dim={embs.shape[1]}")

    # Default scopes
    if args.scope:
        scope_specs = [s.strip() for s in args.scope.split(";") if s.strip()]
    elif args.corpus == "alldot" and "dot" in meta.columns:
        scope_specs = ["document_series", "dot", "dot,document_series"]
    else:
        scope_specs = ["document_series"]

    results = {"corpus": args.corpus, "n_chunks": int(len(meta)),
               "restrict_to": args.restrict_to, "axes": []}

    for spec in scope_specs:
        fields = [f.strip() for f in spec.split(",")]
        missing = [f for f in fields if f not in meta.columns]
        if missing:
            print(f"  [skip] scope '{spec}': missing columns {missing}")
            continue
        labels = _compose(meta, fields)
        axis_name = " x ".join(fields)
        results["axes"].append(analyse_axis(embs, labels, args.k, axis_name))

    out_json = Path(f"{prefix}_dilution{args.tag}.json")
    out_json.write_text(json.dumps(results, indent=2))

    for ax in results["axes"]:
        print(md_table(ax))
    print(f"\n[{args.corpus}] wrote {out_json}")


if __name__ == "__main__":
    main()
