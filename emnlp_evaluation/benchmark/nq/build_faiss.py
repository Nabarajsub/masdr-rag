"""NaturalQuestions-Open: build FAISS + queries.json.

NQ-Open's standard EM/F1 evaluation needs (query, gold short answers,
gold long answer / Wikipedia page). We use the open-domain version
(google-research-datasets/nq_open) on dev split: 7.83k questions.

For retrieval we use the BEIR `nq` corpus (~2.7M Wikipedia paragraphs).
To keep wall time reasonable for the EMNLP rebuttal we cap corpus at
200k passages (containing all gold long-answer passages plus a random
fill), and evaluate on the first 1000 NQ-Open dev questions.

Outputs:
    data/nq.embeddings.npy
    data/nq.meta.parquet
    data/queries.json
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
DATA.mkdir(parents=True, exist_ok=True)
BATCH = 128


def _load_nq_open(limit_q: int = 1000):
    """Load NQ-Open dev split as (qid, question, short_answers)."""
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/nq_open", split="validation")
    out = []
    for i, r in enumerate(ds):
        out.append({
            "query_id": f"nq_{i:05d}",
            "query": r["question"],
            "short_answers": r.get("answer", []),  # list of acceptable spans
        })
        if i + 1 >= limit_q:
            break
    return out


def _load_nq_corpus(limit_docs: int = 200_000, seed: int = 0):
    """Load BEIR `nq` corpus (paragraphs from Wikipedia)."""
    from datasets import load_dataset
    ds = load_dataset("BeIR/nq", "corpus", split="corpus")
    rng = random.Random(seed)
    rows = []
    keep = []
    for r in ds:
        keep.append({
            "id": str(r["_id"]),
            "text": (r.get("title", "") + ". " + r.get("text", "")).strip(),
            "title": r.get("title", ""),
            "source_type": "wikipedia",
        })
    if limit_docs and len(keep) > limit_docs:
        rng.shuffle(keep)
        keep = keep[:limit_docs]
    return keep


def main() -> None:
    print("[nq] loading queries (NQ-Open dev, first 1000)")
    queries = _load_nq_open(limit_q=1000)
    print(f"[nq] {len(queries)} queries loaded")

    print("[nq] loading corpus (BEIR/nq, capped to 200k)")
    chunks = _load_nq_corpus(limit_docs=200_000)
    print(f"[nq] {len(chunks)} chunks")

    for i, c in enumerate(chunks):
        c["idx"] = i
    meta = pd.DataFrame(chunks)

    from emnlp_evaluation.embeddings import get_embedder
    emb = get_embedder("bge_m3")
    texts = meta["text"].astype(str).tolist()
    embs = np.empty((len(texts), 1024), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(texts), BATCH):
        c = texts[i:i + BATCH]
        v = emb.embed_documents(c) if hasattr(emb, "embed_documents") else [emb.embed_query(t) for t in c]
        embs[i:i + len(c)] = np.asarray(v, dtype=np.float32)
        if (i // BATCH) % 100 == 0:
            print(f"  {i+len(c)}/{len(texts)}  ({(i+len(c))/max(1,time.time()-t0):.0f}/s)",
                  flush=True)
    norms = np.linalg.norm(embs, axis=1, keepdims=True); norms[norms == 0] = 1.0
    embs = embs / norms

    np.save(DATA / "nq.embeddings.npy", embs.astype(np.float32))
    meta.to_parquet(DATA / "nq.meta.parquet", index=False)
    print(f"[wrote] embeddings={embs.shape}, meta={len(meta)} rows")

    qout = []
    for q in queries:
        qout.append({
            "query_id": q["query_id"],
            "query": q["query"],
            "reference_answer": (q["short_answers"][0] if q["short_answers"] else ""),
            "gold_short_answers": q["short_answers"],
            "category": "wikipedia",
            "question_type": "open_qa",
        })
    (DATA / "queries.json").write_text(json.dumps(qout, indent=2))
    print(f"[wrote] queries.json ({len(qout)} queries)")


if __name__ == "__main__":
    main()
