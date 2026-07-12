"""
Ingest MultiHop-RAG corpus into Neo4j under separate labels (MultihopDoc,
MultihopChunk) so it co-exists with the WYDOT subgraph.

Chunking: corpus articles are already paragraph-sized; we split anything
longer than ~800 chars by sentences to stay within BGE-M3's effective range.

Usage:
    sbatch slurm/run_multihop_ingest.sbatch        # full ingest on GPU node
    python -m emnlp_evaluation.benchmark.multihop_rag.ingest --limit 100
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

from neo4j import GraphDatabase

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "graph_processing"))

from emnlp_evaluation.configs import oss_config as cfg
from emnlp_evaluation.embeddings import get_embedder


DATA_DIR = Path(__file__).resolve().parent / "data"
MULTIHOP_INDEX = "multihop_bge_m3_index"
MULTIHOP_FULLTEXT_INDEX = "multihop_fulltext"


def _connect():
    return GraphDatabase.driver(
        cfg.NEO4J_URI, auth=(cfg.NEO4J_USERNAME, cfg.NEO4J_PASSWORD),
    )


def _sess(driver):
    kw = {"database": cfg.NEO4J_DATABASE} if cfg.NEO4J_DATABASE else {}
    return driver.session(**kw)


def _chunk_text(text: str, max_len: int = 800) -> List[str]:
    text = (text or "").strip()
    if len(text) <= max_len:
        return [text] if text else []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) > max_len and cur:
            chunks.append(cur.strip()); cur = ""
        cur += s + " "
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def _chunk_id(article_id: str, idx: int) -> str:
    return f"mh::{article_id}::{idx:03d}"


def _ensure_schema(driver, dim: int):
    with _sess(driver) as s:
        s.run("CREATE CONSTRAINT multihop_doc_id IF NOT EXISTS "
              "FOR (m:MultihopDoc) REQUIRE m.id IS UNIQUE")
        s.run("CREATE CONSTRAINT multihop_chunk_id IF NOT EXISTS "
              "FOR (c:MultihopChunk) REQUIRE c.id IS UNIQUE")
        s.run(f"""
            CREATE VECTOR INDEX {MULTIHOP_INDEX} IF NOT EXISTS
            FOR (c:MultihopChunk) ON (c.embedding)
            OPTIONS {{ indexConfig: {{
                `vector.dimensions`: {dim},
                `vector.similarity_function`: 'cosine'
            }} }}""")
        s.run(f"CREATE FULLTEXT INDEX {MULTIHOP_FULLTEXT_INDEX} IF NOT EXISTS "
              f"FOR (c:MultihopChunk) ON EACH [c.text]")


def _iter_articles(corpus_path: Path, limit: int | None) -> Iterable[Dict]:
    with open(corpus_path) as f:
        corpus = json.load(f)
    for i, art in enumerate(corpus):
        if limit and i >= limit:
            return
        yield art


def _article_id(art: Dict) -> str:
    aid = art.get("id") or art.get("url") or art.get("title", "")
    return hashlib.md5(aid.encode("utf-8")).hexdigest()[:16]


def _write_article(driver, art: Dict, chunks: List[Dict]):
    payload = {
        "id": _article_id(art),
        "title": art.get("title", ""),
        "source": art.get("source") or art.get("publisher") or "unknown",
        "author": art.get("author", ""),
        "category": art.get("category", ""),
        "url": art.get("url", ""),
        "published_at": art.get("published_at") or art.get("date", ""),
    }
    with _sess(driver) as s:
        s.run("""
            MERGE (m:MultihopDoc {id: $id})
            SET m.title = $title, m.source = $source, m.author = $author,
                m.category = $category, m.url = $url, m.published_at = $published_at
        """, **payload)
        s.run("""
            UNWIND $rows AS row
            MERGE (c:MultihopChunk {id: row.id})
            SET c.text = row.text, c.idx = row.idx, c.embedding = row.embedding,
                c.source = row.source, c.title = row.title
            WITH c, row
            MATCH (m:MultihopDoc {id: row.doc_id})
            MERGE (m)-[:HAS_CHUNK]->(c)
        """, rows=[{
            "id": c["id"], "text": c["text"], "idx": c["idx"],
            "embedding": c["embedding"],
            "doc_id": payload["id"],
            "source": payload["source"], "title": payload["title"],
        } for c in chunks])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(DATA_DIR / "corpus.json"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    embedder = get_embedder("bge_m3")
    driver = _connect()
    _ensure_schema(driver, dim=embedder.dim)

    t0 = time.time()
    n_articles = n_chunks = 0
    for art in _iter_articles(Path(args.corpus), args.limit):
        n_articles += 1
        aid = _article_id(art)
        text_chunks = _chunk_text(art.get("body") or art.get("text") or "")
        if not text_chunks:
            continue
        embeddings = embedder.embed_documents(text_chunks)
        rows = [
            {"id": _chunk_id(aid, i), "text": t, "idx": i, "embedding": e}
            for i, (t, e) in enumerate(zip(text_chunks, embeddings))
        ]
        _write_article(driver, art, rows)
        n_chunks += len(rows)
        if n_articles % 25 == 0:
            elapsed = time.time() - t0
            print(f"[ingest] {n_articles} articles, {n_chunks} chunks  ({elapsed:.0f}s)", flush=True)

    print(f"[ingest] done. {n_articles} articles / {n_chunks} chunks in {time.time()-t0:.0f}s")
    driver.close()


if __name__ == "__main__":
    main()
