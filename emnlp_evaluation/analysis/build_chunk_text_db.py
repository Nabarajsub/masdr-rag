"""Build chunk_id -> text JSON databases for the LLM-as-judge.

Without these, the judge sees "(chunk texts unavailable)" and scores
faithfulness=0 for every record, which is why all OSS-pipeline
faithfulness scores have been zero. This script materialises one JSON per
corpus:

    composite -> data/composite.meta.parquet           (offline)
    crag      -> data/crag.meta.parquet                (offline)
    wydot     -> production Neo4j (READ ONLY MATCH...) (network)

The WYDOT path is the only one that hits production. It is a read-only
Cypher MATCH; nothing is written. Policy-compliant.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict


def _dump_parquet(prefix: Path, out_path: Path) -> int:
    import pandas as pd
    meta = pd.read_parquet(f"{prefix}.meta.parquet")
    if "id" not in meta.columns or "text" not in meta.columns:
        raise SystemExit(f"missing id/text columns in {prefix}.meta.parquet: {meta.columns.tolist()}")
    db: Dict[str, str] = dict(zip(meta["id"].astype(str), meta["text"].astype(str)))
    out_path.write_text(json.dumps(db))
    return len(db)


def _dump_wydot_neo4j(out_path: Path) -> int:
    from neo4j import GraphDatabase, READ_ACCESS
    from emnlp_evaluation.configs import oss_config as cfg
    driver = GraphDatabase.driver(cfg.NEO4J_URI, auth=(cfg.NEO4J_USERNAME, cfg.NEO4J_PASSWORD))
    db: Dict[str, str] = {}
    session_kw = {"default_access_mode": READ_ACCESS}
    if cfg.NEO4J_DATABASE:
        session_kw["database"] = cfg.NEO4J_DATABASE
    with driver.session(**session_kw) as sess:
        for rec in sess.run("MATCH (n) WHERE n.id IS NOT NULL AND n.text IS NOT NULL "
                            "RETURN n.id AS id, n.text AS text"):
            db[str(rec["id"])] = str(rec["text"] or "")
    driver.close()
    out_path.write_text(json.dumps(db))
    return len(db)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=("composite", "crag", "wydot"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default=None,
                    help="Prefix path for composite/crag parquet (default: built-in).")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # __file__ = .../graph_processing/emnlp_evaluation/analysis/build_chunk_text_db.py
    # parents[2] is graph_processing already.
    repo = Path(__file__).resolve().parents[2]
    if args.corpus == "composite":
        pref = Path(args.prefix or repo / "emnlp_evaluation/composite_corpus/data/composite")
        n = _dump_parquet(pref, out)
    elif args.corpus == "crag":
        pref = Path(args.prefix or repo / "emnlp_evaluation/benchmark/crag/data/crag")
        n = _dump_parquet(pref, out)
    else:
        n = _dump_wydot_neo4j(out)
    print(f"[db] {args.corpus}: wrote {n} chunks -> {out}")


if __name__ == "__main__":
    main()
