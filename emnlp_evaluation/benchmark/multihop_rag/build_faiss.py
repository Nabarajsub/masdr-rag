"""Build FAISS index for MultiHop-RAG, per local-store convention.

Outputs (same shape as composite_corpus.local_backend expects):
    data/multihop.embeddings.npy
    data/multihop.meta.parquet
    data/queries.json  (normalized: query_id, query, reference_answer,
                        category, gold_titles, question_type)

Categories double as routing scopes (business, technology, sports,
entertainment, science, health). Each article is split into ~3-sentence
chunks (~250-500 chars), embedded with BGE-M3, and L2-normalised.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from emnlp_evaluation.embeddings import get_embedder

DATA = Path(__file__).resolve().parent / "data"
DATA.mkdir(exist_ok=True)
BATCH = 128


def _chunk(body: str) -> list[str]:
    sentences = [s.strip() for s in body.replace("\n", " ").split(". ") if s.strip()]
    out = []
    for i in range(0, len(sentences), 3):
        grp = sentences[i:i + 3]
        text = ". ".join(grp).rstrip(".") + "."
        if len(text) >= 50:
            out.append(text)
    return out


def main() -> None:
    raw_q = DATA / "MultiHopRAG.json"
    raw_c = DATA / "corpus.json"
    queries = json.loads(raw_q.read_text())
    corpus = json.loads(raw_c.read_text())
    print(f"[load] {len(queries)} queries  {len(corpus)} articles")

    rows = []
    for i, art in enumerate(corpus):
        body = art.get("body") or ""
        for j, text in enumerate(_chunk(body)):
            rows.append({
                "id": f"mhrag::doc_{i:05d}::chunk_{j:03d}",
                "text": text,
                "source_type": (art.get("category") or "unknown").lower(),
                "title": art.get("title", ""),
                "publisher": art.get("source", ""),
                "published_at": art.get("published_at", ""),
                "doc_id": f"doc_{i:05d}",
                "url": art.get("url", ""),
                "idx": len(rows),
            })
    meta = pd.DataFrame(rows)
    print(f"[chunk] {len(meta)} chunks, categories={sorted(meta['source_type'].unique())}")

    print("[bge] loading embedder")
    emb = get_embedder("bge_m3")
    texts = meta["text"].astype(str).tolist()
    embs = np.empty((len(texts), 1024), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(texts), BATCH):
        chunk = texts[i:i + BATCH]
        vecs = emb.embed_documents(chunk) if hasattr(emb, "embed_documents") \
               else [emb.embed_query(t) for t in chunk]
        embs[i:i + len(chunk)] = np.asarray(vecs, dtype=np.float32)
        if (i // BATCH) % 20 == 0:
            done = i + len(chunk)
            rate = done / max(1, time.time() - t0)
            eta = (len(texts) - done) / max(1, rate)
            print(f"  {done}/{len(texts)}  rate={rate:.0f}/s  eta={eta/60:.1f}min",
                  flush=True)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = embs / norms

    np.save(DATA / "multihop.embeddings.npy", embs.astype(np.float32))
    meta.to_parquet(DATA / "multihop.meta.parquet", index=False)
    print(f"[wrote] embeddings={embs.shape}, meta={len(meta)} rows")

    qout = []
    for q in queries:
        ev = q.get("evidence_list") or []
        gold_titles = sorted({(e.get("title") or "").strip() for e in ev if e.get("title")})
        qout.append({
            "query_id": q.get("query_id") or f"mhq_{len(qout):04d}",
            "query": q.get("query", ""),
            "reference_answer": q.get("answer", ""),
            "question_type": q.get("question_type", ""),
            "category": "cross_domain",
            "gold_titles": gold_titles,
            "n_evidence": len(ev),
        })
    (DATA / "queries.json").write_text(json.dumps(qout, indent=2))
    print(f"[wrote] queries.json ({len(qout)} queries)")


if __name__ == "__main__":
    main()
