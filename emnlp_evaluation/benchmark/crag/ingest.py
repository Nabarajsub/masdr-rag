"""
Flatten CRAG's per-query `search_results` into a corpus parquet + embed it
with BGE-M3 into the LocalSearchBackend layout.

Scoping field is `domain` (finance / sports / music / movie / open).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO.parent))
sys.path.insert(0, str(_REPO))

from emnlp_evaluation.embeddings import get_embedder


DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_PREFIX = DATA_DIR / "crag"


def _hash(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]


def _chunk(text: str, max_chars: int = 800) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    sents = re.split(r"(?<=[.!?])\s+", text)
    out, cur = [], ""
    for s in sents:
        if len(cur) + len(s) > max_chars and cur:
            out.append(cur.strip()); cur = ""
        cur += s + " "
    if cur.strip():
        out.append(cur.strip())
    return out


def _iter_rows(jsonl_path: Path) -> Iterable[Dict]:
    with open(jsonl_path) as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _domain(ex: Dict) -> str:
    return (ex.get("domain") or ex.get("category") or "open").lower()


def _per_paragraph_domain(title: str) -> str:
    """For HotpotQA we treat each first-letter bucket as a scope agent so we
    have a non-trivial number of scope groups. This is a crude proxy but it
    lets the architecture exercise scoping without per-article agents."""
    title = (title or "").strip()
    if not title:
        return "other"
    c = title[0].upper()
    if c.isalpha():
        # Bucket A-G, H-M, N-S, T-Z, other into 5 'topic' buckets.
        if c <= "G": return "topic_ag"
        if c <= "M": return "topic_hm"
        if c <= "S": return "topic_ns"
        return "topic_tz"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=str(DATA_DIR / "crag.jsonl"))
    ap.add_argument("--out", default=str(OUT_PREFIX))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-pages-per-query", type=int, default=5)
    args = ap.parse_args()

    src = Path(args.jsonl)
    if not src.exists():
        sys.exit(f"missing {src}; run benchmark/crag/download.py first.")

    # Collect corpus chunks + queries
    chunks: List[Dict] = []
    queries: List[Dict] = []
    seen_chunks = set()
    for ex in _iter_rows(src):
        q_dom = _domain(ex)
        q_text = ex.get("query") or ex.get("question") or ""
        q_ans = ex.get("answer") or ex.get("ground_truth") or ""
        gold_titles = set(ex.get("gold_titles") or [])
        results = (ex.get("search_results") or ex.get("retrieval_results") or [])[: args.max_pages_per_query]
        doc_ids = []
        gold_doc_ids = []
        # scope_type for a chunk = first-letter-bucket of its article title;
        # gives the architecture a non-trivial scoping signal even though
        # HotpotQA has no native source-type field.
        for j, res in enumerate(results):
            title = res.get("page_name") or res.get("title") or ""
            snippet = res.get("page_snippet") or res.get("snippet") or res.get("page_content") or ""
            full = res.get("page_content") or snippet
            doc_id = _hash(f"{q_text}::{j}::{title}")
            paragraph_dom = _per_paragraph_domain(title)
            for k, ch in enumerate(_chunk(full, max_chars=800)):
                cid = f"crag::{doc_id}::{k:03d}"
                if cid in seen_chunks: continue
                seen_chunks.add(cid)
                chunks.append({
                    "id": cid, "doc_id": doc_id,
                    "source_type": paragraph_dom,
                    "title": title[:120], "text": ch, "idx": k,
                })
            doc_ids.append(doc_id)
            if title in gold_titles:
                gold_doc_ids.append(doc_id)
        queries.append({
            "query_id": f"crag_{_hash(q_text)}",
            "query": q_text,
            "reference_answer": q_ans,
            "category": q_dom,
            "gold_doc_ids": gold_doc_ids,
            "gold_titles": list(gold_titles),
            "question_type": ex.get("question_type"),
            "static_or_dynamic": ex.get("static_or_dynamic"),
        })

    if not chunks:
        sys.exit("no chunks extracted; CRAG file shape unrecognized.")

    df = pd.DataFrame(chunks)
    print(f"[crag-ingest] {len(df)} chunks across {df['source_type'].nunique()} domains")

    # Embed
    embedder = get_embedder("bge_m3")
    vecs = []
    texts = df["text"].astype(str).tolist()
    t0 = time.time()
    for i in range(0, len(texts), args.batch_size):
        batch = texts[i : i + args.batch_size]
        vecs.extend(embedder.embed_documents(batch))
        if (i // args.batch_size) % 20 == 0:
            print(f"[crag-ingest] embedded {i+len(batch)}/{len(texts)} "
                  f"({(i+len(batch))/max(time.time()-t0,1e-6):.1f}/s)", flush=True)
    arr = np.asarray(vecs, dtype=np.float32)

    out = Path(args.out)
    np.save(f"{out}.embeddings.npy", arr)
    df.to_parquet(f"{out}.meta.parquet", index=False)
    queries_out = out.with_suffix(".queries.json")
    with open(queries_out, "w") as f:
        json.dump(queries, f)
    print(f"[crag-ingest] wrote {out}.embeddings.npy, {out}.meta.parquet, {queries_out}")
    print(f"   {len(queries)} queries, {len(df)} chunks")


if __name__ == "__main__":
    main()
