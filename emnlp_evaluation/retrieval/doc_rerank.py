"""Two-stage document-scoped retrieval.

Motivation. Scope routing has a hard ceiling on WYDOT: even an *oracle*
`document_series` only lifts gold-document P@10 from .281 to .381, because the
answer-bearing document still competes with ~20 siblings inside its own series.
Document identity, not scope, is the granularity that separates. Measured on the
32 gold-labelled queries at the full 1,083-document index:

    stage-1 candidate recall@10 docs   1.000
    stage-1 top-1 document             0.438
    + cross-encoder rerank             0.594
    end-to-end P@10 / R@10             .606 / .875   (vs monolithic .281 / .750)

Stages:
  1. Rank documents by the mean of their `topm` best chunk similarities and keep
     the top `n_cand`. Free -- reuses the query-chunk similarities retrieval
     already computes.
  2. Rerank those candidates with a cross-encoder scored per *chunk*, taking each
     document's best chunk. Scoring chunks individually beats concatenating them
     into one blob (top-1 .594 vs .562).
  3. Fuse softly rather than restricting hard: a chunk's score is its similarity
     plus `lam` times its document's normalised rerank score. A wrong document
     pick then degrades to global retrieval instead of to zero -- hard top-1
     gives P@10 .594 but drops R@10 to .594, while soft fusion holds R@10 at .875.

`lam` sits on a broad plateau (1.5-3.0), so the setting is not knife-edge.

CAVEAT: tuned and measured on n=32 gold-labelled queries, WYDOT only. One query
is 3.1 points, so differences of 1-2 queries between variants are not resolvable.
"""
from __future__ import annotations

import numpy as np

DEFAULT_LAM = 2.0


def document_scores(sims: np.ndarray, doc_of_chunk: np.ndarray, n_docs: int,
                    topm: int = 3, pool: int = 5000) -> np.ndarray:
    """Score every document by the mean similarity of its `topm` best chunks."""
    out = np.full(n_docs, -2.0, dtype=np.float32)
    acc: dict[int, list[float]] = {}
    for j in np.argsort(-sims)[:pool]:
        acc.setdefault(int(doc_of_chunk[j]), []).append(float(sims[j]))
    for d, vals in acc.items():
        out[d] = float(np.mean(vals[:topm]))
    return out


def rerank_documents(query, cand_docs, sims, doc_of_chunk, texts, cross_encoder,
                     top_chunks: int = 5, max_chars: int = 1000) -> dict[int, float]:
    """Cross-encode each candidate's best chunks; a document scores as its best chunk."""
    pairs, owner = [], []
    for d in cand_docs:
        idx = np.flatnonzero(doc_of_chunk == d)
        for j in idx[np.argsort(-sims[idx])[:top_chunks]]:
            pairs.append((query, texts[j][:max_chars]))
            owner.append(int(d))
    if not pairs:
        return {}
    scores = cross_encoder.predict(pairs, batch_size=64, show_progress_bar=False)
    best: dict[int, float] = {}
    for d, s in zip(owner, scores):
        best[d] = max(best.get(d, -1e9), float(s))
    return best


def retrieve(query, sims, doc_of_chunk, texts, cross_encoder, *, n_docs,
             k=10, n_cand=10, lam=DEFAULT_LAM, topm=3):
    """Return the indices of the top-`k` chunks under soft document-prior fusion."""
    ds = document_scores(sims, doc_of_chunk, n_docs, topm=topm)
    cand = np.argsort(-ds)[:n_cand]
    rr = rerank_documents(query, cand, sims, doc_of_chunk, texts, cross_encoder)
    if not rr:
        return np.argsort(-sims)[:k]

    vals = np.array([rr[int(d)] for d in cand], dtype=np.float32)
    z = (vals - vals.mean()) / (vals.std() + 1e-9)
    # Documents outside the candidate set inherit the weakest candidate's prior,
    # so they stay reachable -- this is what preserves recall on a bad pick.
    prior = np.full(n_docs, float(z.min()), dtype=np.float32)
    for d, v in zip(cand, z):
        prior[int(d)] = v
    fused = sims + lam * 0.1 * prior[doc_of_chunk]
    return np.argpartition(-fused, k)[:k]
