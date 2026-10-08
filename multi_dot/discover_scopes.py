#!/usr/bin/env python3
"""Adaptive scope discovery: cluster document embeddings into retrieval scopes.

Answers the reviewer criticism that MASDR-RAG "relies on pre-existing document
categories": can the scope taxonomy be recovered (or improved) from embeddings
alone, with no hand metadata?

Method: mean-pool BGE-M3 chunk embeddings per document -> L2-normalize ->
k-means over documents -> assign every chunk its doc's cluster. Evaluate
alignment with the hand `document_series` axis (NMI/ARI/purity, doc- and
chunk-weighted) plus cluster balance. The retrieval-time eval over discovered
scopes is a separate step (run_wydot_oss with a scope-column override).

Usage:
    python -m graph_processing.multi_dot.discover_scopes --corpus wydot
    python -m graph_processing.multi_dot.discover_scopes --corpus wydot --k-sweep 5 20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

REPO = Path(__file__).resolve().parents[2]
SEED = 42


def load(corpus: str):
    sub = "_combined" if corpus == "alldot" else corpus
    prefix = REPO / "data" / sub / "derived" / corpus
    emb = np.load(f"{prefix}.embeddings.npy", mmap_mode="r")
    meta = pd.read_parquet(f"{prefix}.meta.parquet")
    assert len(meta) == emb.shape[0]
    return emb, meta


def doc_pool(emb, meta):
    """Mean-pool chunk embeddings per document, L2-normalized."""
    codes, uniques = pd.factorize(meta["source"])
    dim = emb.shape[1]
    sums = np.zeros((len(uniques), dim), dtype=np.float64)
    counts = np.bincount(codes, minlength=len(uniques)).astype(np.float64)
    # accumulate in blocks to keep the mmap read sequential
    B = 20000
    for lo in range(0, emb.shape[0], B):
        hi = min(lo + B, emb.shape[0])
        np.add.at(sums, codes[lo:hi], np.asarray(emb[lo:hi], dtype=np.float64))
    docvecs = sums / counts[:, None]
    docvecs /= np.linalg.norm(docvecs, axis=1, keepdims=True) + 1e-12
    return docvecs.astype(np.float32), codes, uniques


def purity(labels_true, labels_pred, weights=None):
    df = pd.DataFrame({"t": labels_true, "p": labels_pred})
    df["w"] = 1.0 if weights is None else weights
    top = df.groupby(["p", "t"])["w"].sum().groupby(level=0).max().sum()
    return float(top / df["w"].sum())


def evaluate(k, docvecs, codes, meta, doc_series):
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(docvecs)
    doc_cluster = km.labels_
    chunk_cluster = doc_cluster[codes]
    chunk_series = meta["document_series"].to_numpy()

    chunks_per_cluster = np.bincount(chunk_cluster, minlength=k)
    res = {
        "k": k,
        "doc_nmi": normalized_mutual_info_score(doc_series, doc_cluster),
        "doc_ari": adjusted_rand_score(doc_series, doc_cluster),
        "doc_purity": purity(doc_series, doc_cluster),
        "chunk_nmi": normalized_mutual_info_score(chunk_series, chunk_cluster),
        "chunk_purity": purity(chunk_series, chunk_cluster),
        "cluster_chunk_share": (chunks_per_cluster / chunks_per_cluster.sum()).round(4).tolist(),
        "inertia": float(km.inertia_),
    }
    # same metrics excluding the 'General' grab-bag, where hand labels are weakest
    mask_doc = doc_series != "General"
    mask_chunk = chunk_series != "General"
    res["doc_nmi_no_general"] = normalized_mutual_info_score(
        doc_series[mask_doc], doc_cluster[mask_doc])
    res["doc_purity_no_general"] = purity(doc_series[mask_doc], doc_cluster[mask_doc])
    res["chunk_purity_no_general"] = purity(chunk_series[mask_chunk], chunk_cluster[mask_chunk])
    return res, doc_cluster


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="wydot")
    ap.add_argument("--k-sweep", nargs=2, type=int, metavar=("LO", "HI"))
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    emb, meta = load(args.corpus)
    docvecs, codes, uniques = doc_pool(emb, meta)
    # ground-truth series per doc = modal series of its chunks
    doc_series = (meta.groupby("source")["document_series"]
                  .agg(lambda s: s.mode().iat[0])
                  .reindex(uniques).to_numpy())
    print(f"[{args.corpus}] {emb.shape[0]} chunks, {len(uniques)} docs, "
          f"{meta['document_series'].nunique()} hand series")

    ks = range(args.k_sweep[0], args.k_sweep[1] + 1) if args.k_sweep else [args.k]
    rows, assignments = [], {}
    for k in ks:
        res, doc_cluster = evaluate(k, docvecs, codes, meta, doc_series)
        rows.append(res)
        assignments[k] = doc_cluster
        print(f"  k={k:2d}  docNMI={res['doc_nmi']:.3f}  docPur={res['doc_purity']:.3f}  "
              f"chunkPur={res['chunk_purity']:.3f}  "
              f"chunkPur\\General={res['chunk_purity_no_general']:.3f}")

    out = REPO / "data" / args.corpus / "derived" / f"{args.corpus}_discovered_scopes.json"
    out.write_text(json.dumps({"corpus": args.corpus, "seed": SEED, "results": rows}, indent=2))
    # persist the doc->cluster map for the best-k (highest doc NMI) for downstream retrieval eval
    best = max(rows, key=lambda r: r["doc_nmi"])
    doc_map = pd.DataFrame({"source": uniques, "hand_series": doc_series,
                            "discovered_scope": assignments[best["k"]]})
    map_path = out.with_name(f"{args.corpus}_scope_map_k{best['k']}.parquet")
    doc_map.to_parquet(map_path)
    print(f"\nbest k={best['k']} by doc NMI={best['doc_nmi']:.3f}")
    print(f"wrote {out.name} and {map_path.name}")


if __name__ == "__main__":
    main()
