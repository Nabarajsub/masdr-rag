"""
Re-embed every Chunk in Neo4j with BGE-M3 and write a new vector index.

Run once on a GPU node (sbatch slurm/reindex_bge_m3.sbatch). The new index
co-exists with the existing Gemini index, so production traffic is unaffected.

Usage:
    python -m emnlp_evaluation.embeddings.reindex_neo4j \\
        [--batch-size 64] [--limit 1000] [--dry-run]
"""
from __future__ import annotations

import argparse
import time
from typing import Iterable, List

from neo4j import GraphDatabase

from ..configs import oss_config as cfg
from .factory import get_embedder


def _connect():
    driver = GraphDatabase.driver(
        cfg.NEO4J_URI, auth=(cfg.NEO4J_USERNAME, cfg.NEO4J_PASSWORD),
    )
    return driver


def _session(driver):
    kwargs = {"database": cfg.NEO4J_DATABASE} if cfg.NEO4J_DATABASE else {}
    return driver.session(**kwargs)


def _ensure_index(driver, dim: int):
    """Create vector index for BGE-M3 embeddings if missing."""
    with _session(driver) as sess:
        sess.run(
            f"""
            CREATE VECTOR INDEX {cfg.NEO4J_BGE_M3_INDEX} IF NOT EXISTS
            FOR (c:Chunk) ON (c.{cfg.NEO4J_BGE_M3_PROPERTY})
            OPTIONS {{
              indexConfig: {{
                `vector.dimensions`: {dim},
                `vector.similarity_function`: 'cosine'
              }}
            }}
            """
        )


def _iter_chunks(driver, batch_size: int, limit: int | None) -> Iterable[List[dict]]:
    """Yield chunks lacking the new embedding, in batches."""
    seen = 0
    after_id = ""
    while True:
        with _session(driver) as sess:
            query = f"""
            MATCH (c:Chunk)
            WHERE c.id > $after_id AND c.{cfg.NEO4J_BGE_M3_PROPERTY} IS NULL
            RETURN c.id AS id, c.text AS text
            ORDER BY c.id
            LIMIT $batch
            """
            rows = list(sess.run(query, after_id=after_id, batch=batch_size))
        if not rows:
            return
        batch = [{"id": r["id"], "text": r["text"] or ""} for r in rows]
        yield batch
        after_id = batch[-1]["id"]
        seen += len(batch)
        if limit and seen >= limit:
            return


def _write_embeddings(driver, items: List[dict]):
    with _session(driver) as sess:
        sess.run(
            f"""
            UNWIND $rows AS row
            MATCH (c:Chunk {{id: row.id}})
            SET c.{cfg.NEO4J_BGE_M3_PROPERTY} = row.emb
            """,
            rows=items,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional cap on number of chunks (for smoke tests).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Embed but skip writing to Neo4j.")
    args = ap.parse_args()

    embedder = get_embedder("bge_m3")
    driver = _connect()

    print(f"[reindex] ensuring vector index '{cfg.NEO4J_BGE_M3_INDEX}' (dim={embedder.dim})")
    _ensure_index(driver, dim=embedder.dim)

    total, t0 = 0, time.time()
    for batch in _iter_chunks(driver, args.batch_size, args.limit):
        texts = [b["text"] for b in batch]
        vecs = embedder.embed_documents(texts)
        rows = [{"id": b["id"], "emb": v} for b, v in zip(batch, vecs)]
        if not args.dry_run:
            _write_embeddings(driver, rows)
        total += len(rows)
        elapsed = time.time() - t0
        rate = total / elapsed if elapsed > 0 else 0
        print(f"[reindex] {total} chunks  ({rate:.1f}/s, elapsed {elapsed:.0f}s)", flush=True)

    print(f"[reindex] done. {total} chunks embedded in {time.time()-t0:.0f}s")
    driver.close()


if __name__ == "__main__":
    main()
