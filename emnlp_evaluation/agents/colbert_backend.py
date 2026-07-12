"""
ColBERTv2 late-interaction baseline.

Wraps `ragatouille.RAGPretrainedModel` (which wraps Stanford ColBERTv2) so
its retrieval method satisfies the same SearchBackend protocol as the dense
(BGE-M3) and sparse (BM25) backends. We build a fresh ColBERT index from the
corpus's chunk parquet on first load and reuse it on subsequent runs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..configs import oss_config as cfg


@dataclass
class ColBERTBackend:
    prefix: str                             # parquet prefix (same as LocalSearchBackend)
    index_root: Optional[str] = None        # where ColBERT writes its index
    # Default to the locally patched snapshot (config.json model_type=bert)
    # so the compute node never needs to hit huggingface.co. Override by
    # setting COLBERT_MODEL_PATH. Path is created by colbert_probe.py.
    model_name: str = os.environ.get(
        "COLBERT_MODEL_PATH",
        "<DATA_ROOT>",
    )
    scope_field: str = "source_type"
    source_filters: Dict[str, List[str]] = field(default_factory=dict)
    index_name: str = "colbert_idx"

    _meta: Optional[pd.DataFrame] = None
    _model = None
    _searcher = None
    _id_lookup: Dict[str, int] = field(default_factory=dict)

    def load(self) -> "ColBERTBackend":
        from ragatouille import RAGPretrainedModel

        meta_path = f"{self.prefix}.meta.parquet"
        self._meta = pd.read_parquet(meta_path).reset_index(drop=True)
        self._id_lookup = {row_id: i for i, row_id in enumerate(self._meta["id"].astype(str))}

        if self.index_root is None:
            self.index_root = str(Path(self.prefix).parent / ".ragatouille")
        os.makedirs(self.index_root, exist_ok=True)

        index_path = os.path.join(self.index_root, "colbert", "indexes", self.index_name)
        if os.path.isdir(index_path):
            print(f"[colbert] loading existing index from {index_path}", flush=True)
            self._model = RAGPretrainedModel.from_index(index_path)
        else:
            print(f"[colbert] building index from {len(self._meta)} chunks "
                  f"using {self.model_name}", flush=True)
            self._model = RAGPretrainedModel.from_pretrained(
                self.model_name, index_root=self.index_root,
            )
            self._model.index(
                collection=self._meta["text"].astype(str).tolist(),
                document_ids=self._meta["id"].astype(str).tolist(),
                document_metadatas=[
                    {"source_type": s, "title": t}
                    for s, t in zip(
                        self._meta[self.scope_field].astype(str),
                        self._meta["title"].astype(str),
                    )
                ],
                index_name=self.index_name,
                max_document_length=512,
                split_documents=False,
            )
        return self

    # ── helpers ──────────────────────────────────────────────────────────────

    def _scope_mask(self, agent_name: str) -> Optional[set]:
        allowed = self.source_filters.get(agent_name)
        if not allowed:
            return None
        mask = self._meta[self.scope_field].isin(allowed)
        return set(self._meta.loc[mask, "id"].astype(str).tolist())

    def _row(self, doc_id: str, score: float, kind: str) -> Optional[Dict[str, Any]]:
        idx = self._id_lookup.get(str(doc_id))
        if idx is None:
            return None
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

    def _topk(self, query: str, allowed_ids: Optional[set], limit: int, kind: str) -> List[Dict]:
        # Over-fetch then post-filter by scope.
        k = limit if allowed_ids is None else max(limit * 4, 40)
        hits = self._model.search(query, k=k)
        out: List[Dict] = []
        for h in hits:
            did = str(h.get("document_id") or h.get("id"))
            if allowed_ids is not None and did not in allowed_ids:
                continue
            rec = self._row(did, h.get("score", 0.0), kind)
            if rec is not None:
                out.append(rec)
            if len(out) >= limit:
                break
        return out

    # ── interface ────────────────────────────────────────────────────────────

    def scoped_vector_search(self, query, agent_name, *, year=None, section=None,
                             limit: int = cfg.DEFAULT_SCOPED_LIMIT) -> List[Dict]:
        return self._topk(query, self._scope_mask(agent_name), limit, "colbert_scoped")

    def scoped_fulltext_search(self, query, agent_name, *, year=None,
                               limit: int = cfg.DEFAULT_FULLTEXT_LIMIT) -> List[Dict]:
        return self._topk(query, self._scope_mask(agent_name), limit, "colbert_scoped")

    def combined_scoped_search(self, query, agent_name, *, year=None, section=None,
                                limit: int = cfg.DEFAULT_SCOPED_LIMIT) -> List[Dict]:
        return self._topk(query, self._scope_mask(agent_name), limit, "colbert_scoped")

    def global_search(self, query, *, year=None,
                       limit: int = cfg.DEFAULT_SCOPED_LIMIT) -> List[Dict]:
        return self._topk(query, None, limit, "colbert_global")


def load_colbert(prefix: str | Path, *, scope_field: str = "source_type",
                 source_filters: Optional[Dict[str, List[str]]] = None,
                 index_name: str = "colbert_idx") -> ColBERTBackend:
    return ColBERTBackend(
        prefix=str(prefix),
        scope_field=scope_field,
        source_filters=source_filters or {},
        index_name=index_name,
    ).load()
