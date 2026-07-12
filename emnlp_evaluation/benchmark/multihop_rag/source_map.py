"""
Source-publisher → scoped agent mapping for MultiHop-RAG.

We treat each publisher as one "domain agent" — the same pattern WYDOT uses
with document_series. The actual list of publishers is discovered from the
downloaded corpus at runtime, so this stays correct even if MultiHop-RAG
adds sources later.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List


DATA_DIR = Path(__file__).resolve().parent / "data"


def _slug(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_")
    return s or "unknown"


def discover_sources(corpus_path: Path | None = None) -> List[str]:
    corpus_path = corpus_path or (DATA_DIR / "corpus.json")
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"{corpus_path} not found — run benchmark/multihop_rag/download.py first."
        )
    with open(corpus_path) as f:
        corpus = json.load(f)
    seen: Dict[str, int] = {}
    for art in corpus:
        src = art.get("source") or art.get("publisher") or "unknown"
        seen[src] = seen.get(src, 0) + 1
    return sorted(seen, key=lambda k: -seen[k])


def build_source_filters(sources: List[str]) -> Dict[str, List[str]]:
    """source_name -> list of Cypher WHERE fragments scoping (m:MultihopDoc)."""
    return {
        f"{_slug(src)}_agent": [f"m.source = '{src.replace(chr(39), chr(39)+chr(39))}'"]
        for src in sources
    }
