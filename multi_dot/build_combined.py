#!/usr/bin/env python3
"""Phase 4: assemble the merged 3-DOT mega-corpus.

Concatenates the per-DOT BGE-M3 outputs into one corpus so dilution can be
measured across DOTs (the headline experiment). No re-embedding: BGE-M3 vectors
are already in a shared space.

  data/_combined/derived/alldot.meta.parquet   (carries `dot` + `document_series`)
  data/_combined/derived/alldot.embeddings.npy

Usage:
    python -m graph_processing.multi_dot.build_combined
    python -m graph_processing.multi_dot.build_combined --dots wydot caltrans cdot
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dots", nargs="+", default=["wydot", "caltrans", "cdot"])
    args = ap.parse_args()

    metas, embs = [], []
    for dot in args.dots:
        d = REPO / "data" / dot / "derived"
        mp, ep = d / f"{dot}.meta.parquet", d / f"{dot}.embeddings.npy"
        if not mp.exists() or not ep.exists():
            raise SystemExit(f"missing {dot} outputs: {mp} / {ep} "
                             f"(run classify_embed for {dot} first)")
        m = pd.read_parquet(mp)
        e = np.load(ep).astype(np.float32)
        if len(m) != len(e):
            raise SystemExit(f"{dot}: meta/emb length mismatch {len(m)} vs {len(e)}")
        print(f"  {dot:10s} {len(m):8d} chunks")
        metas.append(m)
        embs.append(e)

    meta = pd.concat(metas, ignore_index=True)
    emb = np.concatenate(embs, axis=0)
    assert len(meta) == len(emb)

    out = REPO / "data" / "_combined" / "derived"
    out.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(out / "alldot.meta.parquet", index=False)
    np.save(out / "alldot.embeddings.npy", emb)

    print(f"\n[alldot] {len(meta)} chunks across {meta['dot'].nunique()} DOTs, "
          f"dim={emb.shape[1]}")
    print(f"[alldot] -> {out/'alldot.meta.parquet'}")
    print("[alldot] chunks per DOT x series:")
    pivot = meta.groupby(["dot", "document_series"]).size().unstack(fill_value=0)
    print(pivot.to_string())


if __name__ == "__main__":
    main()
