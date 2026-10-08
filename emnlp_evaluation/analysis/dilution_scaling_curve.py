"""E19: causal dilution curve -- hold the query set and gold evidence FIXED,
grow the distractor pool, measure P@10.

This is the controlled version of the deployment observation in the paper's
opening ("scaling from 54 to 1,128 documents cut accuracy"). That figure was an
operational anecdote with no protocol; this measures the same phenomenon with a
fixed query set, fixed gold documents, and a single varying factor (corpus size).

P@10 here is gold-DOCUMENT membership: a retrieved chunk counts as relevant if it
comes from one of the query's gold_titles. That is stricter than the scope-purity
P@10 used elsewhere in the paper, and it is the metric a reviewer would ask for
("does retrieval still surface the answer-bearing document?").

Usage:
    python -m emnlp_evaluation.analysis.dilution_scaling_curve \
        --prefix ../data/wydot/derived/wydotv3 --seeds 5 --out results.json
"""
from __future__ import annotations
import argparse, ast, json, random
from pathlib import Path

import numpy as np
import pandas as pd


def load(prefix: str):
    meta = pd.read_parquet(prefix + ".meta.parquet")
    emb = np.load(prefix + ".embeddings.npy", mmap_mode="r")
    queries = json.load(open(prefix + ".queries.json"))
    scorable = []
    for r in queries:
        gt = r.get("gold_titles")
        if isinstance(gt, str):
            gt = ast.literal_eval(gt)
        scope = r.get("gold_hand_scope")
        if gt:
            scorable.append({"qid": r["query_id"], "query": r["query"],
                             "gold": [str(t) for t in gt], "scope": str(scope) if scope else None,
                             "category": r.get("category"), "mode": "doc"})
        elif scope and scope != "General":
            scorable.append({"qid": r["query_id"], "query": r["query"],
                             "gold": None, "scope": str(scope),
                             "category": r.get("category"), "mode": "scope"})
    return meta, emb, scorable


def embed_queries(texts):
    from emnlp_evaluation.embeddings import get_embedder
    enc = get_embedder("bge_m3")
    V = np.asarray(enc.embed_documents(texts), dtype=np.float32)
    V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    return V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--sizes", default="16,54,100,200,400,700,1083")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--mode", default="doc", choices=("doc","scope"))
    ap.add_argument("--retrieval", default="global", choices=("global","oracle_scope"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    meta, emb, qs = load(a.prefix)
    qs = [q for q in qs if q["mode"] == a.mode] if a.mode == "doc" else \
         [q for q in qs if q.get("scope")]
    titles = meta.title.astype(str).values
    series = meta.document_series.astype(str).values
    doc_series = dict(zip(titles, series))
    print(f"[data] {len(meta)} chunks, {len(set(titles))} docs, {len(qs)} scorable queries")

    if a.mode == "doc":
        gold_docs = sorted({t for q in qs for t in q["gold"]})
    else:
        gold_docs = []                      # scope mode uses stratified sampling
    distractors = sorted(set(titles) - set(gold_docs))
    print(f"[data] {len(gold_docs)} gold docs, {len(distractors)} distractors")

    Q = embed_queries([q["query"] for q in qs])
    E = np.array(emb, dtype=np.float32)   # copy: emb is a read-only mmap
    E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    S = Q @ E.T                       # (nq, nchunks) -- computed once
    print(f"[sim] {S.shape}")

    out = []
    for size in [int(x) for x in a.sizes.split(",")]:
        n_extra = max(0, size - len(gold_docs))
        for seed in range(a.seeds):
            rng = random.Random(1000 + seed)
            if a.mode == "doc":
                keep = set(gold_docs) | set(rng.sample(distractors, min(n_extra, len(distractors))))
            else:
                # stratified by document_series so every scope stays represented
                bys = {}
                for t in set(titles): bys.setdefault(doc_series[t], []).append(t)
                keep = set()
                for sname, docs in sorted(bys.items()):
                    n_s = max(1, round(size * len(docs) / len(set(titles))))
                    keep |= set(rng.sample(sorted(docs), min(n_s, len(docs))))
            sel = np.isin(titles, list(keep))
            idx = np.flatnonzero(sel)
            per_q = []; hits_doc = []
            for i, q in enumerate(qs):
                cand = idx
                if a.retrieval == "oracle_scope" and q.get("scope"):
                    cand = idx[series[idx] == q["scope"]]
                    if len(cand) == 0:
                        per_q.append(0.0); continue
                sims = S[i, cand]
                top = cand[np.argpartition(-sims, min(a.k, len(sims) - 1))[:a.k]]
                if a.mode == "doc":
                    gset = set(q["gold"])
                    hit = sum(titles[j] in gset for j in top)
                else:
                    hit = sum(series[j] == q["scope"] for j in top)
                per_q.append(hit / a.k)
                hits_doc.append(1.0 if hit > 0 else 0.0)
            out.append({"size": size, "n_docs": len(keep), "seed": seed,
                        "p_at_10": float(np.mean(per_q)),
                        "r_at_10_doc": float(np.mean(hits_doc)),
                        "n_chunks": int(sel.sum())})
            print(f"  size={size:5d} docs={len(keep):5d} chunks={sel.sum():7d} "
                  f"seed={seed} P@10={np.mean(per_q):.4f} R@10doc={np.mean(hits_doc):.4f}")
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\n[done] wrote {a.out}")
    # summary
    df = pd.DataFrame(out).groupby("size")[["p_at_10","r_at_10_doc"]].agg(["mean","std"])
    print(df.to_string())


if __name__ == "__main__":
    main()
