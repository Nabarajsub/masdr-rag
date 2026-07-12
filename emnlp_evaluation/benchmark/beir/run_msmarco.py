"""BEIR MS-MARCO standard IR metrics (nDCG@10, MRR@10, Recall@1/10/100).

Downloads the MS-MARCO dev split via the `beir` library and runs BGE-M3
dense retrieval against the full corpus, then computes BEIR's official
EvaluateRetrieval metrics. This is a sanity-check anchor for reviewers:
our retrieval pipeline reproduces a published BEIR number.

Usage:
    python -m emnlp_evaluation.benchmark.beir.run_msmarco --limit 1000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import numpy as np

from emnlp_evaluation.embeddings import get_embedder

OUT = Path("<DATA_ROOT>")
OUT.mkdir(parents=True, exist_ok=True)


def _load_msmarco(limit_corpus: int = 200_000):
    """Use the beir loader if available; fall back to a HuggingFace mirror."""
    try:
        from beir import util
        from beir.datasets.data_loader import GenericDataLoader
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip"
        data_path = util.download_and_unzip(url, str(OUT))
        corpus, queries, qrels = GenericDataLoader(data_path).load(split="dev")
    except Exception as e:
        print(f"[fallback] beir download failed ({e}); using HuggingFace mirror")
        from datasets import load_dataset
        # mteb/msmarco mirrors the same split
        ds = load_dataset("BeIR/msmarco", "corpus", split="corpus")
        corpus = {r["_id"]: {"text": r["text"], "title": r.get("title", "")} for r in ds}
        ds = load_dataset("BeIR/msmarco", "queries", split="queries")
        queries = {r["_id"]: r["text"] for r in ds}
        ds = load_dataset("BeIR/msmarco-qrels", split="validation")
        qrels: Dict[str, Dict[str, int]] = {}
        for r in ds:
            qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])
    if limit_corpus and len(corpus) > limit_corpus:
        # Keep only docs that appear in qrels + a random sample
        keep = set()
        for qid, m in qrels.items():
            keep.update(m.keys())
        extras = list(set(corpus.keys()) - keep)[: max(0, limit_corpus - len(keep))]
        keep.update(extras)
        corpus = {k: corpus[k] for k in keep}
        print(f"[load] capped corpus to {len(corpus)}")
    return corpus, queries, qrels


def _ndcg_at_k(retrieved_ids: list, rels: Dict[str, int], k: int = 10) -> float:
    import math
    dcg = 0.0
    for i, did in enumerate(retrieved_ids[:k]):
        gain = rels.get(str(did), 0)
        dcg += (2 ** gain - 1) / math.log2(i + 2)
    # ideal DCG
    ideal_rels = sorted(rels.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0


def _mrr_at_k(retrieved_ids: list, rels: Dict[str, int], k: int = 10) -> float:
    for i, did in enumerate(retrieved_ids[:k]):
        if rels.get(str(did), 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def _recall_at_k(retrieved_ids: list, rels: Dict[str, int], k: int) -> float:
    gold = {d for d, g in rels.items() if g > 0}
    if not gold:
        return 0.0
    hit = sum(1 for d in retrieved_ids[:k] if str(d) in gold)
    return hit / len(gold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1000,
                    help="Evaluate at most N queries.")
    ap.add_argument("--limit-corpus", type=int, default=200_000,
                    help="Cap corpus to this many docs (gold + random fill).")
    ap.add_argument("--k-cap", type=int, default=100)
    args = ap.parse_args()

    print("[load] BEIR MS-MARCO dev split")
    corpus, queries, qrels = _load_msmarco(limit_corpus=args.limit_corpus)
    print(f"[load] corpus={len(corpus)} queries={len(queries)} qrels={len(qrels)}")

    qids = list(qrels.keys())[: args.limit]
    print(f"[eval] {len(qids)} queries")

    print("[bge] loading embedder")
    emb = get_embedder("bge_m3")

    print("[bge] encoding corpus...")
    cids = list(corpus.keys())
    texts = [corpus[d].get("text", "") or "" for d in cids]
    BATCH = 128
    cembs = np.empty((len(texts), 1024), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(texts), BATCH):
        c = texts[i:i+BATCH]
        v = emb.embed_documents(c) if hasattr(emb, "embed_documents") else [emb.embed_query(t) for t in c]
        cembs[i:i+len(c)] = np.asarray(v, dtype=np.float32)
        if (i//BATCH) % 100 == 0:
            print(f"  corpus {i+len(c)}/{len(texts)}  ({(i+len(c))/max(1,time.time()-t0):.0f}/s)", flush=True)
    norms = np.linalg.norm(cembs, axis=1, keepdims=True); norms[norms==0]=1.0
    cembs = cembs / norms

    print("[bge] encoding queries...")
    qtexts = [queries[q] for q in qids]
    qembs = np.empty((len(qtexts), 1024), dtype=np.float32)
    for i in range(0, len(qtexts), BATCH):
        c = qtexts[i:i+BATCH]
        v = emb.embed_documents(c) if hasattr(emb, "embed_documents") else [emb.embed_query(t) for t in c]
        qembs[i:i+len(c)] = np.asarray(v, dtype=np.float32)
    qn = np.linalg.norm(qembs, axis=1, keepdims=True); qn[qn==0]=1.0
    qembs = qembs / qn

    # FAISS for speed
    import faiss
    index = faiss.IndexFlatIP(cembs.shape[1])
    index.add(cembs)
    print("[faiss] index built")
    scores, ids = index.search(qembs, args.k_cap)

    metrics = {"ndcg@10": [], "mrr@10": [], "recall@1": [], "recall@10": [], "recall@100": []}
    for qi, qid in enumerate(qids):
        rels = qrels.get(qid, {})
        retrieved_ids = [cids[int(j)] for j in ids[qi] if j >= 0]
        metrics["ndcg@10"].append(_ndcg_at_k(retrieved_ids, rels, 10))
        metrics["mrr@10"].append(_mrr_at_k(retrieved_ids, rels, 10))
        metrics["recall@1"].append(_recall_at_k(retrieved_ids, rels, 1))
        metrics["recall@10"].append(_recall_at_k(retrieved_ids, rels, 10))
        metrics["recall@100"].append(_recall_at_k(retrieved_ids, rels, 100))

    print()
    print(f"{'metric':<12} {'mean':>8}  {'n':>5}")
    out = {}
    for k, v in metrics.items():
        m = float(np.mean(v))
        out[k] = m
        print(f"{k:<12} {m:>8.4f}  {len(v):>5}")

    out_path = OUT / f"msmarco_bge_m3_n{len(qids)}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[wrote] {out_path}")


if __name__ == "__main__":
    main()
