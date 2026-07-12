"""S-5: build a unified-FAISS WYDOT index from production Neo4j (READ ONLY).

Reads (id, text, source, year, section, doc_id) for every WYDOT chunk
using a single READ_ACCESS MATCH, embeds with BGE-M3 on GPU, writes the
LocalSearchBackend layout (<prefix>.embeddings.npy + <prefix>.meta.parquet)
so the existing run_composite / run_wydot harness can swap Neo4j for a
fully local FAISS-only store.

Outputs:
    emnlp_evaluation/composite_corpus/data/wydot.embeddings.npy
    emnlp_evaluation/composite_corpus/data/wydot.meta.parquet
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from neo4j import GraphDatabase, READ_ACCESS

from emnlp_evaluation.configs import oss_config as cfg
from emnlp_evaluation.embeddings import get_embedder

OUT = Path("<DATA_ROOT>")
OUT.parent.mkdir(parents=True, exist_ok=True)

BATCH = 256


def main() -> None:
    print(f"[neo4j] read from {cfg.NEO4J_URI[:50]}...")
    driver = GraphDatabase.driver(cfg.NEO4J_URI, auth=(cfg.NEO4J_USERNAME, cfg.NEO4J_PASSWORD))
    skw = {"default_access_mode": READ_ACCESS}
    if cfg.NEO4J_DATABASE:
        skw["database"] = cfg.NEO4J_DATABASE
    rows = []
    with driver.session(**skw) as sess:
        # Pull chunk + its document context so the local store has the same
        # filterable fields (source, year, section) the Neo4j path used.
        cypher = """
        MATCH (d:Document)-[:HAS_SECTION]->(s:Section)-[:HAS_CHUNK]->(n)
        WHERE n.id IS NOT NULL AND n.text IS NOT NULL
        RETURN n.id AS id, n.text AS text,
               coalesce(d.source, '') AS source,
               coalesce(d.title,  '') AS title,
               coalesce(d.year,   '') AS year,
               coalesce(s.name,   '') AS section,
               coalesce(d.id,     '') AS doc_id
        """
        for rec in sess.run(cypher):
            rows.append(dict(rec))
    driver.close()
    print(f"[neo4j] {len(rows)} chunks pulled")

    meta = pd.DataFrame(rows)
    # LocalSearchBackend reads scope_field='source_type' by default; keep
    # 'source' as that column for direct compatibility.
    meta.rename(columns={"source": "source_type"}, inplace=True)
    meta["idx"] = range(len(meta))

    print("[bge] loading embedder")
    emb = get_embedder("bge_m3")
    texts = meta["text"].astype(str).tolist()
    print(f"[bge] encoding {len(texts)} chunks in batches of {BATCH}")
    embs = np.empty((len(texts), 1024), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(texts), BATCH):
        chunk = texts[i:i + BATCH]
        # Most embedders expose .embed_documents; fall back to embed_query
        if hasattr(emb, "embed_documents"):
            vecs = emb.embed_documents(chunk)
        else:
            vecs = [emb.embed_query(t) for t in chunk]
        embs[i:i + len(chunk)] = np.asarray(vecs, dtype=np.float32)
        if (i // BATCH) % 20 == 0:
            done = i + len(chunk)
            rate = done / max(1, time.time() - t0)
            eta = (len(texts) - done) / max(1, rate)
            print(f"  {done}/{len(texts)}  rate={rate:.0f}/s  eta={eta/60:.1f}min", flush=True)

    # L2-normalize so IndexFlatIP behaves like cosine.
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = embs / norms

    np.save(f"{OUT}.embeddings.npy", embs.astype(np.float32))
    meta.to_parquet(f"{OUT}.meta.parquet", index=False)
    print(f"[wrote] {OUT}.embeddings.npy  ({embs.shape})")
    print(f"[wrote] {OUT}.meta.parquet  ({len(meta)} rows)")


if __name__ == "__main__":
    main()
