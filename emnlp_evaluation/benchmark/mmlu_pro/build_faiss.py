"""MMLU-Pro: cross-subject QA — 14 subjects, perfect for scoping experiments.

Each question carries a subject (math, physics, biology, business, etc.)
and 10 multiple-choice options. We treat the subject as the routing
scope. For retrieval-RAG we use the question stem + options as the
chunk content (no separate corpus, so this acts as a "self-retrieval"
benchmark — the question chunk IS the gold passage).

Outputs:
    data/mmlu_pro.embeddings.npy
    data/mmlu_pro.meta.parquet
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
    print("[mmlu] loading TIGER-Lab/MMLU-Pro")
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test",
                      cache_dir="<DATA_ROOT>")
    print(f"[mmlu] {len(ds)} test records")

    # Cap to 1000 questions for the rebuttal sweep.
    LIMIT = 1000
    queries = []
    chunks = []
    for i, r in enumerate(ds.shuffle(seed=0).select(range(min(LIMIT, len(ds))))):
        qid = f"mmlu_{i:04d}"
        subject = (r.get("category") or "general").lower().replace(" ", "_")
        # Chunk text: the options form a self-contained context.
        opts = r.get("options") or []
        opt_str = "\n".join(f"({chr(65+j)}) {o}" for j, o in enumerate(opts))
        text = f"Subject: {r.get('category','')}\nQuestion: {r['question']}\n{opt_str}"
        cid = f"mmlu::{subject}::{i}"
        chunks.append({
            "id": cid, "text": text, "source_type": subject,
            "title": r.get("category", ""), "question_id": r.get("question_id", i),
            "doc_id": cid, "idx": i,
        })
        ans_idx = r.get("answer_index", -1)
        gold_letter = chr(65 + int(ans_idx)) if ans_idx >= 0 else ""
        gold_ans = opts[int(ans_idx)] if 0 <= int(ans_idx) < len(opts) else r.get("answer", "")
        queries.append({
            "query_id": qid,
            "query": r["question"],
            "reference_answer": f"({gold_letter}) {gold_ans}",
            "category": subject,
            "gold_chunk_id": cid,
            "options": opts,
            "answer_index": ans_idx,
            "question_type": "mcq",
        })

    print(f"[mmlu] {len(queries)} queries, {len(chunks)} self-chunks")
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

    np.save(DATA / "mmlu_pro.embeddings.npy", embs.astype(np.float32))
    meta.to_parquet(DATA / "mmlu_pro.meta.parquet", index=False)
    (DATA / "queries.json").write_text(json.dumps(queries, indent=2))
    print(f"[wrote] {DATA}/mmlu_pro.*  ({len(meta)} chunks, {len(queries)} queries)")
    print(f"[mmlu] subjects: {sorted(set(c['source_type'] for c in chunks))}")


if __name__ == "__main__":
    main()
