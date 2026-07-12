"""
BM25-backed SearchBackend.

Provides the exact same interface as LocalSearchBackend / SearchBackend so
the orchestrator / hybrid / ReAct / naive runners can plug it in unchanged.
Uses `rank_bm25` (Okapi BM25) over an in-memory parquet metadata file.

This is the external-system baseline EMNLP reviewers expect: a classical
sparse retriever side-by-side with the dense retrievers (Gemini, BGE-M3,
ColBERT).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..configs import oss_config as cfg


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


@dataclass
class BM25Backend:
    prefix: str                                # parquet metadata prefix (same as LocalSearchBackend)
    scope_field: str = "source_type"
    source_filters: Dict[str, List[str]] = field(default_factory=dict)

    _meta: Optional[pd.DataFrame] = None
    _bm25 = None
    _tokens: Optional[List[List[str]]] = None

    def load(self) -> "BM25Backend":
        from rank_bm25 import BM25Okapi
        meta_path = f"{self.prefix}.meta.parquet"
        self._meta = pd.read_parquet(meta_path).reset_index(drop=True)
        self._tokens = [_tokenize(t) for t in self._meta["text"].astype(str)]
        self._bm25 = BM25Okapi(self._tokens)
        return self

    # ── scoping ──
    def _scope_mask(self, agent_name: str) -> np.ndarray:
        allowed = self.source_filters.get(agent_name)
        if not allowed:
            return np.ones(len(self._meta), dtype=bool)
        return self._meta[self.scope_field].isin(allowed).to_numpy()

    def _row_to_dict(self, idx: int, score: float, kind: str) -> Dict[str, Any]:
        row = self._meta.iloc[idx]
        return {
            "id": str(row["id"]),
            "text": str(row.get("text", "")),
            "source": str(row.get(self.scope_field, "")),
            "title": str(row.get("title", "")),
            "year": row.get("year") if "year" in row else None,
            "section": str(row.get("section", "")),
            "page": int(row.get("idx", 0)),
            "score": float(score),
            "search_type": kind,
        }

    # ── interface methods ──
    def _topk(self, query: str, mask: np.ndarray, limit: int, kind: str) -> List[Dict]:
        if not mask.any():
            return []
        toks = _tokenize(query)
        if not toks:
            return []
        scores = self._bm25.get_scores(toks)
        if not mask.all():
            scores = scores * mask  # zero out non-scope rows
        top = np.argsort(-scores)[:limit]
        return [self._row_to_dict(int(i), float(scores[i]), kind)
                for i in top if scores[i] > 0]

    def scoped_vector_search(self, query, agent_name, *, year=None, section=None,
                             limit: int = cfg.DEFAULT_SCOPED_LIMIT) -> List[Dict]:
        # BM25 has no dense vector; we just call the same path for both APIs.
        return self._topk(query, self._scope_mask(agent_name), limit, "bm25_scoped")

    def scoped_fulltext_search(self, query, agent_name, *, year=None,
                               limit: int = cfg.DEFAULT_FULLTEXT_LIMIT) -> List[Dict]:
        return self._topk(query, self._scope_mask(agent_name), limit, "bm25_scoped")

    def combined_scoped_search(self, query, agent_name, *, year=None, section=None,
                                limit: int = cfg.DEFAULT_SCOPED_LIMIT) -> List[Dict]:
        return self._topk(query, self._scope_mask(agent_name), limit, "bm25_scoped")

    def global_search(self, query, *, year=None,
                       limit: int = cfg.DEFAULT_SCOPED_LIMIT) -> List[Dict]:
        return self._topk(query, np.ones(len(self._meta), dtype=bool), limit, "bm25_global")


def load_bm25(prefix: str | Path, *, scope_field: str = "source_type",
              source_filters: Optional[Dict[str, List[str]]] = None) -> BM25Backend:
    return BM25Backend(
        prefix=str(prefix),
        scope_field=scope_field,
        source_filters=source_filters or {},
    ).load()
