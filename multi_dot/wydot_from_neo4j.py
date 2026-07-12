#!/usr/bin/env python3
"""Phase 2 (WYDOT): pull the existing WYDOT corpus out of Neo4j.

The WYDOT documents and chunks already live in Neo4j (the production AuraDB)
with the schema  (:Document)-[:HAS_SECTION]->(:Section)-[:HAS_CHUNK]->(:Chunk).
This script reads them out and writes them in the SAME format the Caltrans /
CDOT pipeline produces, so all three DOTs are processed identically downstream:

  data/wydot/derived/chunks.parquet

WYDOT's free-text `document_series` is normalised onto the shared 10-bucket
taxonomy so the cross-DOT dilution comparison is apples-to-apples.

Then re-embed with BGE-M3 (so WYDOT shares the OSS embedding space):
  python -m graph_processing.multi_dot.classify_embed --dot wydot --skip-classify

Run on a login node (AuraDB needs outbound network).

Usage:
    python -m graph_processing.multi_dot.wydot_from_neo4j
    python -m graph_processing.multi_dot.wydot_from_neo4j --max-docs 250
"""
from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from neo4j import GraphDatabase

REPO = Path(__file__).resolve().parents[2]

# Connection: env first, then the values already committed in ingestneo4j_updated.py.
URI = os.getenv("NEO4J_URI_GEMINI") or os.getenv("NEO4J_URI") \
    or "<NEO4J_URI>"
USER = os.getenv("NEO4J_USERNAME_GEMINI") or os.getenv("NEO4J_USERNAME") or "<NEO4J_INSTANCE>"
PASSWORD = os.getenv("NEO4J_PASSWORD_GEMINI") or os.getenv("NEO4J_PASSWORD") \
    or "<NEO4J_PASSWORD>"
# Aura's home database for this instance is the instance id (see agentic_solution/config.py).
DATABASE = os.getenv("NEO4J_DATABASE_GEMINI") or os.getenv("NEO4J_DATABASE") or "<NEO4J_INSTANCE>"

# Normalise WYDOT's free-text document_series onto the shared taxonomy.
NORMALISE_RULES = [
    ("Standard Plans", ["standard plan"]),
    ("Standard Specs", ["standard spec", "specification"]),
    ("Construction Manual", ["construction manual"]),
    ("Materials Testing", ["material", "testing manual"]),
    ("Bridge Program", ["bridge", "approach slab"]),
    ("Design Manual", ["design manual", "design guide", "road design"]),
    ("Traffic & Safety", ["traffic crash", "highway safety", "traffic",
                          "vulnerable road", "safety"]),
    ("STIP", ["transportation improvement", "stip", "corridor", "long range"]),
    ("Annual Reports", ["annual report", "financial", "operating budget"]),
]


def normalise_series(raw: str, title: str) -> str:
    hay = f"{raw or ''}  {title or ''}".lower()
    for bucket, keys in NORMALISE_RULES:
        if any(k in hay for k in keys):
            return bucket
    return "General"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-docs", type=int, default=0,
                    help="stratified sample N docs per category (0 = all)")
    ap.add_argument("--no-rechunk", action="store_true",
                    help="keep the original Neo4j SemanticChunker chunks (not"
                         " recommended for cross-DOT comparison)")
    ap.add_argument("--uri", default=URI)
    args = ap.parse_args()

    print(f"[wydot] connecting to {args.uri}")
    driver = GraphDatabase.driver(args.uri, auth=(USER, PASSWORD))
    rows = []
    with driver.session(database=DATABASE) as sess:
        q = """
        MATCH (d:Document)-[:HAS_SECTION]->(s:Section)-[:HAS_CHUNK]->(c:Chunk)
        RETURN coalesce(c.id, elementId(c))      AS cid,
               c.text                             AS text,
               coalesce(c.page, 0)                AS page,
               coalesce(c.seq, 0)                 AS seq,
               d.source                           AS source,
               coalesce(d.display_title, d.source) AS title,
               coalesce(toString(d.year), '')     AS year,
               coalesce(d.document_series, 'General') AS series,
               coalesce(s.name, 'General')        AS section
        """
        result = sess.run(q)
        for r in result:
            if not r["text"]:
                continue
            rows.append({
                "id": f"wydot::{r['cid']}",
                "dot": "wydot",
                "source": r["source"] or "unknown.pdf",
                "document_series": normalise_series(r["series"], r["title"]),
                "raw_series": r["series"],
                "title": r["title"] or "",
                "year": int(r["year"]) if str(r["year"]).isdigit() else None,
                "section": r["section"] or "General",
                "page": int(r["page"]) if r["page"] else 0,
                "seq": int(r["seq"]) if r["seq"] else 0,
                "text": r["text"],
            })
    driver.close()
    print(f"[wydot] pulled {len(rows)} chunks from Neo4j")
    if not rows:
        raise SystemExit("no chunks returned — check the Neo4j schema/credentials")

    df = pd.DataFrame(rows)
    # A chunk can sit under multiple Sections -> the MATCH path duplicates it.
    before = len(df)
    df = df.drop_duplicates(subset="id").reset_index(drop=True)
    if len(df) != before:
        print(f"[wydot] de-duplicated {before - len(df)} repeated chunk rows")

    # ── Re-chunk WYDOT uniformly so all 3 DOTs share the same chunking ──
    # The original WYDOT chunks come from a SemanticChunker pipeline -- median
    # 1273 chars but max 172,713 chars. Caltrans/CDOT are RecursiveCharacter-
    # TextSplitter(1000, 100). For a fair cross-DOT dilution comparison every
    # corpus must use the same chunking.
    if not args.no_rechunk:
        print("[wydot] re-chunking with RecursiveCharacterTextSplitter(1000, 100) "
              "to match Caltrans / CDOT...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        new_rows = []
        for source, sub in df.groupby("source"):
            sub = sub.sort_values("seq")
            doc_text = "\n".join(sub["text"].astype(str).tolist())
            meta = sub.iloc[0]
            pieces = splitter.split_text(doc_text)
            for i, p in enumerate(pieces):
                if len(p.strip()) < 40:
                    continue
                new_rows.append({
                    "id": f"wydot::{source}::{i}",
                    "dot": "wydot",
                    "source": source,
                    "document_series": meta["document_series"],
                    "title": meta["title"],
                    "year": meta["year"],
                    "section": meta["section"],
                    "page": int(meta["page"] or 0),
                    "text": p,
                })
        df = pd.DataFrame(new_rows)
        print(f"[wydot] re-chunked into {len(df)} chunks "
              f"(was {before - (before-len(df))})")

    # Optional stratified sampling to PoC scale (sample by document, not chunk).
    if args.max_docs > 0:
        by_cat = defaultdict(list)
        for src, sub in df.groupby("source"):
            by_cat[sub["document_series"].iloc[0]].append(src)
        keep = set()
        rng = random.Random(42)
        for cat, srcs in by_cat.items():
            rng.shuffle(srcs)
            keep.update(srcs[:args.max_docs])
        df = df[df["source"].isin(keep)].reset_index(drop=True)
        print(f"[wydot] sampled to {df['source'].nunique()} docs / {len(df)} chunks")

    derived = REPO / "data" / "wydot" / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    df.to_parquet(derived / "chunks.parquet", index=False)
    print(f"[wydot] wrote {derived/'chunks.parquet'}")
    print(f"[wydot] documents={df['source'].nunique()}  chunks={len(df)}")
    print("[wydot] document_series distribution:")
    for series, cnt in df["document_series"].value_counts().items():
        ndocs = df[df.document_series == series]["source"].nunique()
        print(f"  {series:22s} {ndocs:5d} docs  {cnt:7d} chunks")


if __name__ == "__main__":
    main()
