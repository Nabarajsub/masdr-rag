#!/usr/bin/env python3
"""Phase 3/4: ingest a DOT corpus into the local Neo4j sandbox.

Reads data/<sub>/derived/<corpus>.meta.parquet + .embeddings.npy and writes
:<Corpus>Chunk nodes with a per-corpus vector + fulltext index. Idempotent
(re-running drops the corpus's nodes/indexes first).

Each DOT is an isolated node label -- functionally a per-DOT database, the same
isolation mechanism emnlp_evaluation already uses for its 6 corpora. The merged
'alldot' corpus carries both `dot` and `document_series` so the retrieval-time
two-axis dilution experiment can scope on either.

Requires the local Neo4j daemon (emnlp_evaluation/slurm/run_local_neo4j.sbatch);
endpoint auto-loaded by oss_config from /cluster/.../local/neo4j_endpoint.

Usage:
    python -m graph_processing.multi_dot.ingest_dot --corpus cdot
    python -m graph_processing.multi_dot.ingest_dot --corpus alldot
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from neo4j import GraphDatabase

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "graph_processing"))

from emnlp_evaluation.configs import oss_config as cfg          # noqa: E402

BATCH = 500
PROPS = ["id", "text", "title", "year", "section",
         "dot", "document_series", "source", "page"]


def prefix_for(corpus: str) -> Path:
    sub = "_combined" if corpus == "alldot" else corpus
    return REPO / "data" / sub / "derived" / corpus


def driver():
    if not cfg.NEO4J_LOCAL_URI:
        raise SystemExit("NEO4J_LOCAL_URI not set — start the local Neo4j daemon "
                         "(emnlp_evaluation/slurm/run_local_neo4j.sbatch).")
    return GraphDatabase.driver(
        cfg.NEO4J_LOCAL_URI,
        auth=(cfg.NEO4J_LOCAL_USERNAME, cfg.NEO4J_LOCAL_PASSWORD))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True,
                    help="wydot | caltrans | cdot | alldot")
    args = ap.parse_args()

    corpus = args.corpus.lower()
    label = f"{corpus.capitalize()}Chunk"
    vec_index = f"{corpus}_vec"
    ft_index = f"{corpus}_ft"

    prefix = prefix_for(corpus)
    meta = pd.read_parquet(f"{prefix}.meta.parquet")
    embs = np.load(f"{prefix}.embeddings.npy").astype(np.float32)
    if len(meta) != len(embs):
        raise SystemExit(f"meta/emb mismatch: {len(meta)} vs {len(embs)}")
    dim = int(embs.shape[1])
    for col in PROPS:
        if col not in meta.columns:
            meta[col] = None
    print(f"[{corpus}] {len(meta)} chunks, dim={dim} -> :{label}")

    db_kw = {"database": cfg.NEO4J_LOCAL_DATABASE} if cfg.NEO4J_LOCAL_DATABASE else {}
    drv = driver()
    with drv.session(**db_kw) as s:
        # idempotent reset
        s.run(f"DROP INDEX {vec_index} IF EXISTS").consume()
        s.run(f"DROP INDEX {ft_index} IF EXISTS").consume()
        while True:
            n = s.run(f"MATCH (c:{label}) WITH c LIMIT 5000 "
                      f"DETACH DELETE c RETURN count(c) AS n").single()["n"]
            if not n:
                break
            print(f"  [drop] removed {n}")
        s.run(f"CREATE VECTOR INDEX {vec_index} IF NOT EXISTS "
              f"FOR (c:{label}) ON (c.embedding) "
              f"OPTIONS {{ indexConfig: {{ `vector.dimensions`: {dim}, "
              f"`vector.similarity_function`: 'cosine' }} }}").consume()
        s.run(f"CREATE FULLTEXT INDEX {ft_index} IF NOT EXISTS "
              f"FOR (c:{label}) ON EACH [c.text]").consume()

        t0, n = time.time(), len(meta)
        for i in range(0, n, BATCH):
            rows = []
            for j in range(i, min(i + BATCH, n)):
                r = meta.iloc[j]
                yr = r["year"]
                rows.append({
                    "id": str(r["id"]),
                    "text": str(r["text"] or ""),
                    "title": str(r["title"] or ""),
                    "year": int(yr) if pd.notna(yr) and str(yr).strip() not in ("", "None") else None,
                    "section": str(r["section"] or ""),
                    "dot": str(r["dot"] or ""),
                    "document_series": str(r["document_series"] or ""),
                    "source": str(r["source"] or ""),
                    "page": int(r["page"]) if pd.notna(r["page"]) else 0,
                    "emb": [float(x) for x in embs[j]],
                })
            s.run(
                f"UNWIND $rows AS r CREATE (c:{label}) SET "
                f"c.id=r.id, c.text=r.text, c.title=r.title, c.year=r.year, "
                f"c.section=r.section, c.dot=r.dot, "
                f"c.document_series=r.document_series, c.source=r.source, "
                f"c.page=r.page, c.embedding=r.emb",
                rows=rows).consume()
            if (i // BATCH) % 20 == 0:
                done = i + len(rows)
                rate = done / max(1e-6, time.time() - t0)
                print(f"  {done}/{n}  {rate:.0f}/s", flush=True)
        print(f"[{corpus}] ingested {n} chunks in {(time.time()-t0)/60:.1f} min")
    drv.close()

    summary = {"corpus": corpus, "label": label, "vector_index": vec_index,
               "fulltext_index": ft_index, "n_chunks": int(len(meta)),
               "dim": dim}
    out = Path(f"{prefix}_neo4j_ingest.json")
    out.write_text(json.dumps(summary, indent=2))
    print(f"[{corpus}] wrote {out}")


if __name__ == "__main__":
    main()
