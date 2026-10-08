#!/usr/bin/env python3
"""Caltrans thread (final_v3): can unsupervised scope discovery recover the
scoping GRANULARITY that resolves a corpus's dilution, without being told?

The dilution anomaly: on Caltrans the hand `document_series` axis shows no
size→dilution correlation (rho=-0.10), because Caltrans documents are omnibus
(2,374 chunks/doc); the `section` axis (finer, sub-document) resolves it
(rho=-0.845). The question: does chunk-level k-means recover section-level
structure — i.e. would adaptive scoping automatically pick the right grain?

We do NOT measure the dilution factor on the discovered clusters — that is
circular (clusters are embedding-compact by construction, so kNN purity is
trivially ~1). Instead we measure ALIGNMENT between the discovered clusters and
two INDEPENDENT, metadata-derived axes (section, document_series). Alignment is
non-circular: section labels come from document parsing, not the embeddings.

Reports, over a k-sweep, both directions of information:
  homogeneity(true=axis | pred=cluster)  — are clusters pure w.r.t. the axis?
  NMI(cluster, axis)                      — mutual information, size-adjusted

Usage:
    python -m graph_processing.multi_dot.adaptive_granularity --corpus caltrans --k-sweep 50 500
    python -m graph_processing.multi_dot.adaptive_granularity --corpus wydot --k-sweep 10 40
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    homogeneity_score, normalized_mutual_info_score,
)

REPO = Path(__file__).resolve().parents[2]
SEED = 42


def load(corpus):
    p = REPO / "data" / corpus / "derived" / corpus
    emb = np.load(f"{p}.embeddings.npy", mmap_mode="r")
    meta = pd.read_parquet(f"{p}.meta.parquet")
    return emb, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="caltrans")
    ap.add_argument("--k-sweep", nargs=2, type=int, default=[50, 500])
    ap.add_argument("--steps", type=int, default=8)
    args = ap.parse_args()

    emb, meta = load(args.corpus)
    X = np.array(emb, dtype=np.float32)   # copy: mmap is read-only
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    axes = {}
    for col in ("document_series", "section"):
        if col in meta.columns and meta[col].notna().any():
            axes[col] = meta[col].astype(str).to_numpy()
    print(f"[{args.corpus}] {len(meta)} chunks; axes: "
          + ", ".join(f"{k}({len(set(v))})" for k, v in axes.items()))

    ks = np.unique(np.linspace(args.k_sweep[0], args.k_sweep[1], args.steps).astype(int))
    rows = []
    for k in ks:
        km = MiniBatchKMeans(n_clusters=int(k), random_state=SEED,
                             batch_size=4096, n_init=3, max_iter=100).fit(X)
        cl = km.labels_
        rec = {"k": int(k)}
        line = f"  k={k:>4}"
        for name, lab in axes.items():
            rec[f"nmi_{name}"] = normalized_mutual_info_score(lab, cl)
            rec[f"homog_{name}"] = homogeneity_score(lab, cl)  # cluster purity w.r.t. axis
            line += f"   {name}: NMI={rec[f'nmi_{name}']:.3f} homog={rec[f'homog_{name}']:.3f}"
        rows.append(rec)
        print(line)

    # Verdict: at the k that best matches the working axis, does the discovered
    # clustering align with section MORE than with document_series?
    # Verdict uses NMI (size-adjusted), not homogeneity (which rises mechanically
    # with k for any labeling). At the finest k, which independent axis does the
    # unsupervised clustering carry more information about?
    if "section" in axes and "document_series" in axes:
        fin = rows[-1]
        print(f"\n  at k={fin['k']}:  NMI(section)={fin['nmi_section']:.3f}  "
              f"NMI(document_series)={fin['nmi_document_series']:.3f}  "
              f"ratio={fin['nmi_section']/max(fin['nmi_document_series'],1e-9):.2f}x")
        verdict = ("aligns with SECTION (the finer axis)"
                   if fin["nmi_section"] > fin["nmi_document_series"] + 0.05
                   else "aligns with DOCUMENT_SERIES (the coarse axis)")
        print(f"  VERDICT: unsupervised discovery {verdict}")

    out = REPO / "data" / args.corpus / "derived" / f"{args.corpus}_granularity.json"
    out.write_text(json.dumps({"corpus": args.corpus, "seed": SEED, "rows": rows}, indent=2))
    print(f"  wrote {out.name}")


if __name__ == "__main__":
    main()
