"""SPLADE-style sparse-learned retrieval backend.

Uses a MaskedLM head (opensearch-project/opensearch-neural-sparse-encoding-v2-distill)
with SPLADE pooling:  s = max_t log(1 + ReLU(W h_t))
The result is a sparse vector over the vocab (30k dims, ~50-150 nnz per text).
Retrieval scores are dot products between the query and corpus sparse vectors.

Implements the same LocalSearchBackend API so the existing runners can swap it
in via --backend splade. Per-corpus chunk sparse vectors are precomputed and
cached at <prefix>.splade.npz so query-time retrieval is fast.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp


_SPLADE_MODEL = "opensearch-project/opensearch-neural-sparse-encoding-v2-distill"


def _resolve_local_snapshot(model_id: str) -> str:
    hub = Path(os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub"
    folder = hub / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if folder.exists():
        snaps = sorted(folder.iterdir())
        if snaps:
            return str(snaps[0])
    return model_id


class SpladeEncoder:
    """Wraps a MaskedLM with SPLADE pooling."""
    _instance: Optional["SpladeEncoder"] = None

    def __init__(self, model_name: str = _SPLADE_MODEL, device: Optional[str] = None):
        import torch
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        path = _resolve_local_snapshot(model_name)
        print(f"[splade] loading {model_name} (resolved: {path}) on {self.device}", flush=True)
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForMaskedLM.from_pretrained(path)
        self.model.eval()
        self.model.to(self.device)
        self.vocab_size = int(self.model.config.vocab_size)

    @classmethod
    def get(cls) -> "SpladeEncoder":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def encode(self, texts: List[str], *, max_length: int = 256,
               batch_size: int = 32) -> sp.csr_matrix:
        import torch
        rows: list = []
        cols: list = []
        data: list = []
        n_done = 0
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tok(batch, return_tensors="pt", truncation=True,
                            max_length=max_length, padding=True).to(self.device)
            with torch.no_grad():
                out = self.model(**enc).logits      # (B, T, V)
            # SPLADE pooling: log(1 + ReLU(logits)) max over tokens, masked.
            sparse = torch.log1p(torch.relu(out))
            mask = enc["attention_mask"].unsqueeze(-1).bool()
            sparse = sparse.masked_fill(~mask, 0.0)
            scores = sparse.max(dim=1).values        # (B, V)
            for b in range(scores.shape[0]):
                vec = scores[b]
                nz = torch.nonzero(vec, as_tuple=True)[0]
                if len(nz) == 0:
                    continue
                rows.extend([n_done + b] * len(nz))
                cols.extend(nz.cpu().tolist())
                data.extend(vec[nz].cpu().tolist())
            n_done += scores.shape[0]
        return sp.csr_matrix(
            (data, (rows, cols)),
            shape=(len(texts), self.vocab_size),
            dtype=np.float32,
        )


@dataclass
class SpladeBackend:
    prefix: str                              # path prefix, expects <prefix>.meta.parquet + <prefix>.splade.npz
    scope_field: str = "source_type"
    source_filters: Dict[str, List[str]] = field(default_factory=dict)
    _meta: Optional[pd.DataFrame] = None
    _matrix: Optional[sp.csr_matrix] = None
    _encoder: Optional[SpladeEncoder] = None

    def load(self) -> "SpladeBackend":
        self._meta = pd.read_parquet(f"{self.prefix}.meta.parquet").reset_index(drop=True)
        npz = np.load(f"{self.prefix}.splade.npz")
        self._matrix = sp.csr_matrix(
            (npz["data"], npz["indices"], npz["indptr"]),
            shape=tuple(npz["shape"]),
        )
        self._encoder = SpladeEncoder.get()
        assert self._matrix.shape[0] == len(self._meta), "corpus length mismatch"
        return self

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
            "score": float(score),
            "search_type": kind,
        }

    def _topk_sparse(self, q_vec: sp.csr_matrix, mask: np.ndarray, limit: int) -> List[Dict]:
        if not mask.any():
            return []
        # Dot product of query (1, V) with corpus subset rows.
        idx = np.where(mask)[0]
        sub = self._matrix[idx]                # (n_sub, V)
        scores = np.asarray((sub @ q_vec.T).todense()).ravel()
        top = np.argsort(-scores)[:limit]
        return [
            self._row_to_dict(int(idx[i]), float(scores[i]), "splade")
            for i in top
        ]

    def scoped_vector_search(self, query, agent_name, *, year=None, section=None,
                             limit: int = 15):
        q = self._encoder.encode([query])
        return self._topk_sparse(q, self._scope_mask(agent_name), limit)

    def scoped_fulltext_search(self, query, agent_name, *, year=None, limit: int = 10):
        # SPLADE is itself a sparse lexical-ish retriever; reuse scoped_vector_search.
        return self.scoped_vector_search(query, agent_name, limit=limit)

    def combined_scoped_search(self, query, agent_name, *, year=None, section=None,
                                limit: int = 15):
        return self.scoped_vector_search(query, agent_name, year=year, section=section, limit=limit)

    def global_search(self, query, *, year=None, limit: int = 15):
        q = self._encoder.encode([query])
        return self._topk_sparse(q, np.ones(len(self._meta), dtype=bool), limit)


def load_splade_backend(prefix: str, source_filters: Optional[Dict[str, List[str]]] = None,
                        scope_field: str = "source_type") -> SpladeBackend:
    return SpladeBackend(
        prefix=str(prefix), scope_field=scope_field,
        source_filters=source_filters or {},
    ).load()


def build_corpus_splade(prefix: str, batch_size: int = 32, max_length: int = 256) -> None:
    """Encode every text in <prefix>.meta.parquet to SPLADE; save <prefix>.splade.npz."""
    meta = pd.read_parquet(f"{prefix}.meta.parquet")
    texts = meta["text"].astype(str).tolist()
    encoder = SpladeEncoder.get()
    print(f"[splade] encoding {len(texts)} chunks", flush=True)
    t0 = time.time()
    mat = encoder.encode(texts, batch_size=batch_size, max_length=max_length)
    print(f"[splade] done in {(time.time()-t0)/60:.1f} min  shape={mat.shape}  nnz/row={mat.nnz/mat.shape[0]:.0f}")
    np.savez_compressed(f"{prefix}.splade.npz",
                        data=mat.data, indices=mat.indices, indptr=mat.indptr,
                        shape=np.asarray(mat.shape))
    print(f"[splade] wrote {prefix}.splade.npz")
