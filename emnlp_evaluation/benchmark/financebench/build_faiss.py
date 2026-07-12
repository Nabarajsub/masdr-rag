"""FinanceBench: SEC filings + financial-analyst questions.

The dataset is published on HF as PatronusAI/financebench. Each question
includes a `doc_name` (the SEC filing it references) and an `evidence`
text snippet. We chunk the evidence + linked SEC PDFs (when included) and
build a small FAISS index. The corpus is small (~150 docs × ~50 chunks)
so this is a 5-min build.

Outputs:
    data/financebench.embeddings.npy
    data/financebench.meta.parquet
    data/queries.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
DATA.mkdir(parents=True, exist_ok=True)
BATCH = 128


def main() -> None:
    from datasets import load_dataset
    print("[fb] loading PatronusAI/financebench")
    try:
        ds = load_dataset("PatronusAI/financebench", split="train",
                          cache_dir="<DATA_ROOT>")
    except Exception:
        # Public mirror via PatronusGreatAI/financebench_v1
        ds = load_dataset("PatronusGreatAI/financebench_v1", split="train",
                          cache_dir="<DATA_ROOT>")
    print(f"[fb] {len(ds)} records")

    # Each record has: question, answer, evidence (str or list), doc_name, company, sector
    queries = []
    chunks = []
    seen_chunks = set()
    for i, r in enumerate(ds):
        qid = f"fb_{i:04d}"
        company = r.get("company") or r.get("doc_name", "unknown")
        sector = (r.get("sector") or r.get("doc_name") or "unknown").lower().replace(" ", "_")
        evid = r.get("evidence") or r.get("justification") or ""
        if isinstance(evid, list):
            evid = "\n\n".join(str(e) for e in evid)
        # Chunk evidence into ~3-sentence pieces.
        sentences = [s.strip() for s in str(evid).replace("\n", " ").split(". ") if s.strip()]
        ev_chunk_ids = []
        for j in range(0, len(sentences), 3):
            grp = sentences[j:j + 3]
            text = ". ".join(grp).rstrip(".") + "."
            if len(text) < 30:
                continue
            cid = f"fb::doc_{company}::ev_{i}_{j}"
            if cid in seen_chunks:
                continue
            seen_chunks.add(cid)
            chunks.append({
                "id": cid, "text": text, "source_type": sector,
                "title": r.get("doc_name", ""), "company": company,
                "doc_id": r.get("doc_name", ""),
            })
            ev_chunk_ids.append(cid)
        queries.append({
            "query_id": qid,
            "query": r.get("question", ""),
            "reference_answer": r.get("answer", ""),
            "category": sector,
            "gold_chunk_id": ev_chunk_ids[0] if ev_chunk_ids else "",
            "gold_chunk_ids": ev_chunk_ids,
            "question_type": "financial_analysis",
        })

    print(f"[fb] {len(queries)} queries, {len(chunks)} chunks")
    if not chunks:
        raise SystemExit("FinanceBench: zero chunks extracted — check evidence field schema.")

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
    norms = np.linalg.norm(embs, axis=1, keepdims=True); norms[norms == 0] = 1.0
    embs = embs / norms

    np.save(DATA / "financebench.embeddings.npy", embs.astype(np.float32))
    meta.to_parquet(DATA / "financebench.meta.parquet", index=False)
    (DATA / "queries.json").write_text(json.dumps(queries, indent=2))
    print(f"[wrote] {DATA}/financebench.* ({len(meta)} chunks, {len(queries)} queries)")


if __name__ == "__main__":
    main()
