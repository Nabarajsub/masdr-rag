"""
In-memory SearchBackend backed by a FAISS index + parquet metadata.

This replaces Neo4j for the EnterpriseComposite-9 and CRAG benchmarks.
Everything stays on the local filesystem; no AuraDB calls.

Layout on disk (any prefix):
    <prefix>.embeddings.npy   # float32 (N, dim), L2-normalized
    <prefix>.meta.parquet     # columns: id, text, source_type, title, doc_id, idx

API mirrors `agents.tools_oss.SearchBackend`:
    combined_scoped_search, scoped_vector_search, scoped_fulltext_search,
    global_search
so orchestrator / hybrid / ReAct work unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..configs import oss_config as cfg
from ..embeddings.base import Embedder


@dataclass
class LocalSearchBackend:
    embedder: Embedder
    prefix: str                             # path prefix to load <prefix>.embeddings.npy + .meta.parquet
    scope_field: str = "source_type"        # column in meta to filter on
    source_filters: Dict[str, List[str]] = field(default_factory=dict)  # agent -> list of allowed scope values

    # populated by .load()
    _embeddings: Optional[np.ndarray] = None
    _meta: Optional[pd.DataFrame] = None
    _index = None  # optional faiss.IndexFlatIP

    def load(self) -> "LocalSearchBackend":
        emb_path = f"{self.prefix}.embeddings.npy"
        meta_path = f"{self.prefix}.meta.parquet"
        self._embeddings = np.load(emb_path).astype(np.float32)
        self._meta = pd.read_parquet(meta_path).reset_index(drop=True)
        assert len(self._embeddings) == len(self._meta), \
            f"embedding/meta row count mismatch: {len(self._embeddings)} vs {len(self._meta)}"
        try:
            import faiss
            self._index = faiss.IndexFlatIP(self._embeddings.shape[1])
            self._index.add(self._embeddings)
        except ImportError:
            self._index = None
        return self

    # ── scoping helpers ─────────────────────────────────────────────────────

    def _scope_mask(self, agent_name: str) -> np.ndarray:
        allowed = self.source_filters.get(agent_name)
        if not allowed:
            return np.ones(len(self._meta), dtype=bool)
        return self._meta[self.scope_field].isin(allowed).to_numpy()

    def _row_to_dict(self, idx: int, score: float, search_type: str) -> Dict[str, Any]:
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
            "search_type": search_type,
        }

    # ── public API ──────────────────────────────────────────────────────────

    def _vector_topk(self, q_vec: np.ndarray, mask: np.ndarray, limit: int) -> List[Dict]:
        if not mask.any():
            return []
        if self._index is not None and mask.all():
            scores, ids = self._index.search(q_vec[None, :], k=limit)
            return [
                self._row_to_dict(int(i), float(s), "local_vector")
                for s, i in zip(scores[0], ids[0]) if i >= 0
            ]
        # masked path: matmul on the subset
        sub_idx = np.where(mask)[0]
        sub_emb = self._embeddings[sub_idx]
        scores = sub_emb @ q_vec
        top = np.argsort(-scores)[:limit]
        return [
            self._row_to_dict(int(sub_idx[i]), float(scores[i]), "local_vector")
            for i in top
        ]

    def scoped_vector_search(self, query: str, agent_name: str, *,
                             year=None, section=None,
                             limit: int = cfg.DEFAULT_SCOPED_LIMIT) -> List[Dict]:
        q_vec = np.asarray(self.embedder.embed_query(query), dtype=np.float32)
        return self._vector_topk(q_vec, self._scope_mask(agent_name), limit)

    def scoped_fulltext_search(self, query: str, agent_name: str, *,
                               year=None, limit: int = cfg.DEFAULT_FULLTEXT_LIMIT) -> List[Dict]:
        """Cheap BM25-ish fallback: term overlap on masked rows."""
        terms = [t.lower() for t in query.split() if len(t) > 2]
        if not terms:
            return []
        mask = self._scope_mask(agent_name)
        sub_idx = np.where(mask)[0]
        scores = np.zeros(len(sub_idx), dtype=np.float32)
        # to_numpy() on a pandas string series returns object-dtype, which
        # np.char.count cannot accept. Force a fixed-width unicode view.
        texts = self._meta.iloc[sub_idx]["text"].astype(str).str.lower().to_numpy()
        texts_u = np.asarray(texts, dtype=np.str_)
        for term in terms:
            scores += np.char.count(texts_u, term).astype(np.float32)
        top = np.argsort(-scores)[:limit]
        return [
            self._row_to_dict(int(sub_idx[i]), float(scores[i]), "local_fulltext")
            for i in top if scores[i] > 0
        ]

    def combined_scoped_search(self, query: str, agent_name: str, *,
                                year=None, section=None,
                                limit: int = cfg.DEFAULT_SCOPED_LIMIT) -> List[Dict]:
        vec = self.scoped_vector_search(query, agent_name, limit=limit)
        ft = self.scoped_fulltext_search(query, agent_name, limit=max(5, limit // 2))
        seen = {r["id"] for r in vec}
        out = list(vec)
        for r in ft:
            if r["id"] not in seen:
                seen.add(r["id"]); out.append(r)
        return out[:limit]

    def global_search(self, query: str, *, year=None,
                      limit: int = cfg.DEFAULT_SCOPED_LIMIT) -> List[Dict]:
        q_vec = np.asarray(self.embedder.embed_query(query), dtype=np.float32)
        return self._vector_topk(q_vec, np.ones(len(self._meta), dtype=bool), limit)


def load_backend(prefix: str | Path, embedder: Embedder, *,
                  scope_field: str = "source_type",
                  source_filters: Optional[Dict[str, List[str]]] = None) -> LocalSearchBackend:
    return LocalSearchBackend(
        embedder=embedder,
        prefix=str(prefix),
        scope_field=scope_field,
        source_filters=source_filters or {},
    ).load()
