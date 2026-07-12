"""Ingest any LocalSearchBackend-style corpus into the local Neo4j sandbox.

Reads <prefix>.meta.parquet + <prefix>.embeddings.npy and writes
:Document/:<Corpus>Chunk nodes plus a vector + fulltext index per corpus.
Idempotent: re-running drops the corpus's existing nodes/indexes first.

The local Neo4j daemon must be running (see slurm/run_local_neo4j.sbatch);
endpoint info is auto-loaded by oss_config.py from /cluster/.../neo4j_endpoint.

Usage:
    python -m emnlp_evaluation.composite_corpus.ingest_to_local_neo4j \\
        --corpus multihop \\
        --prefix emnlp_evaluation/benchmark/multihop_rag/data/multihop
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from neo4j import GraphDatabase

from emnlp_evaluation.configs import oss_config as cfg


BATCH_W = 200  # Cypher write batch


def _driver():
    if not cfg.NEO4J_LOCAL_URI:
        raise SystemExit("NEO4J_LOCAL_URI not set; the local Neo4j daemon must be running.")
    return GraphDatabase.driver(cfg.NEO4J_LOCAL_URI,
                                auth=(cfg.NEO4J_LOCAL_USERNAME, cfg.NEO4J_LOCAL_PASSWORD))


def _drop_corpus(sess, chunk_label: str, vector_index: str, fulltext_index: str) -> None:
    print(f"[drop] {chunk_label} nodes and indexes (if exist)")
    sess.run(f"DROP INDEX {vector_index} IF EXISTS").consume()
    sess.run(f"DROP INDEX {fulltext_index} IF EXISTS").consume()
    # delete in batches of 5k to avoid 1GB tx limit
    n = 1
    while n > 0:
        rec = sess.run(
            f"MATCH (n:{chunk_label}) WITH n LIMIT 5000 DETACH DELETE n RETURN count(n) AS n"
        ).single()
        n = rec["n"]
        if n:
            print(f"  deleted {n} chunks ...")


def _create_indexes(sess, chunk_label: str, vector_index: str,
                    fulltext_index: str, dim: int) -> None:
    print(f"[index] CREATE VECTOR INDEX {vector_index} on {chunk_label}(embedding)")
    sess.run(
        f"CREATE VECTOR INDEX {vector_index} IF NOT EXISTS "
        f"FOR (n:{chunk_label}) ON (n.embedding) "
        f"OPTIONS {{ indexConfig: {{ "
        f"`vector.dimensions`: {dim}, "
        f"`vector.similarity_function`: 'cosine' "
        f"}} }}"
    ).consume()
    print(f"[index] CREATE FULLTEXT INDEX {fulltext_index} on {chunk_label}(text)")
    sess.run(
        f"CREATE FULLTEXT INDEX {fulltext_index} IF NOT EXISTS "
        f"FOR (n:{chunk_label}) ON EACH [n.text]"
    ).consume()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True,
                    help="Short slug used in node labels and index names "
                         "(e.g. multihop, financebench, mmlu_pro, nq, wydotfaiss).")
    ap.add_argument("--prefix", required=True,
                    help="Path prefix; expects <prefix>.meta.parquet and <prefix>.embeddings.npy.")
    ap.add_argument("--scope-field", default="source_type")
    args = ap.parse_args()

    corpus = args.corpus.lower()
    chunk_label = f"{corpus.capitalize()}Chunk"
    vector_index = f"{corpus}_chunk_vec"
    fulltext_index = f"{corpus}_chunk_ft"

    meta_path = Path(f"{args.prefix}.meta.parquet")
    emb_path = Path(f"{args.prefix}.embeddings.npy")
    if not meta_path.exists() or not emb_path.exists():
        raise SystemExit(f"missing files: {meta_path} / {emb_path}")

    meta = pd.read_parquet(meta_path)
    embs = np.load(emb_path)
    if len(meta) != len(embs):
        raise SystemExit(f"meta/emb length mismatch: {len(meta)} vs {len(embs)}")
    dim = int(embs.shape[1])
    print(f"[load] {len(meta)} chunks, dim={dim}")

    if args.scope_field not in meta.columns:
        print(f"[warn] scope_field '{args.scope_field}' not in meta; using empty string")
        meta[args.scope_field] = ""

    # Ensure required columns
    for col in ("id", "text", "title", "year", "section"):
        if col not in meta.columns:
            meta[col] = "" if col != "year" else None
    meta["text"] = meta["text"].fillna("").astype(str)
    meta["id"] = meta["id"].astype(str)

    driver = _driver()
    db_kw = {"database": cfg.NEO4J_LOCAL_DATABASE} if cfg.NEO4J_LOCAL_DATABASE else {}
    with driver.session(**db_kw) as sess:
        _drop_corpus(sess, chunk_label, vector_index, fulltext_index)
        _create_indexes(sess, chunk_label, vector_index, fulltext_index, dim)

        # Bulk insert via UNWIND batches.
        n = len(meta)
        t0 = time.time()
        for i in range(0, n, BATCH_W):
            batch = []
            for j in range(i, min(i + BATCH_W, n)):
                row = meta.iloc[j]
                batch.append({
                    "id": str(row["id"]),
                    "text": str(row["text"]),
                    "title": str(row.get("title", "")),
                    "year": row.get("year") if pd.notna(row.get("year")) else None,
                    "section": str(row.get("section", "")),
                    "scope": str(row.get(args.scope_field, "")),
                    "emb": [float(x) for x in embs[j]],
                })
            cypher = (
                f"UNWIND $rows AS r "
                f"CREATE (c:{chunk_label} {{ "
                f"  id: r.id, text: r.text, title: r.title, "
                f"  year: r.year, section: r.section, "
                f"  {args.scope_field}: r.scope, embedding: r.emb "
                f"}})"
            )
            sess.run(cypher, rows=batch).consume()
            if (i // BATCH_W) % 25 == 0:
                done = i + len(batch)
                rate = done / max(1, time.time() - t0)
                eta = (n - done) / max(1, rate)
                print(f"  {done}/{n}  rate={rate:.0f}/s  eta={eta/60:.1f}min", flush=True)
        print(f"[done] {n} chunks ingested in {(time.time()-t0)/60:.1f}min")

    driver.close()
    summary = {
        "corpus": corpus,
        "chunk_label": chunk_label,
        "vector_index": vector_index,
        "fulltext_index": fulltext_index,
        "n_chunks": int(len(meta)),
        "embedding_dim": dim,
    }
    out = Path(args.prefix).parent / f"{corpus}_neo4j_ingest.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"[wrote] {out}")


if __name__ == "__main__":
    main()
