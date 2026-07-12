"""Cross-encoder re-rank wrapper backend.

Wraps any backend that implements the LocalSearchBackend interface and
adds a BGE-reranker-v2-m3 cross-encoder pass between the bi-encoder
retrieval and the LLM synthesis. The reranker re-scores the top-K
candidates and returns the top-N (default N=10) sorted by cross-encoder
score.

Pipeline:
    bi-encoder top-K (K=30 default)  ->  cross-encoder rerank  ->  top-N (N=10)

Drop-in replacement: anywhere we wrote
    backend = load_backend(...)
we can write
    backend = RerankBackend(load_backend(...))
and every downstream call (global_search / combined_scoped_search /
scoped_vector_search) gets reranked.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


_RERANKER_SINGLETON = None


def _resolve_local_snapshot(model_id: str) -> str:
    """Return the absolute path to the cached HuggingFace snapshot dir, so
    CrossEncoder loaders can skip the online metadata fetch on offline
    compute nodes."""
    import os
    from pathlib import Path
    hub = Path(os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub"
    folder = hub / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if folder.exists():
        snaps = sorted(folder.iterdir())
        if snaps:
            return str(snaps[0])
    return model_id  # fall back to slug; CrossEncoder will try online


def _get_reranker(model_name: str = "BAAI/bge-reranker-v2-m3"):
    """Lazy-load the cross-encoder; share one instance across all calls."""
    global _RERANKER_SINGLETON
    if _RERANKER_SINGLETON is None:
        from sentence_transformers import CrossEncoder
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        resolved = _resolve_local_snapshot(model_name)
        print(f"[rerank] loading {model_name} (resolved: {resolved}) on {device}", flush=True)
        _RERANKER_SINGLETON = CrossEncoder(resolved, max_length=512, device=device)
    return _RERANKER_SINGLETON


@dataclass
class RerankBackend:
    base: Any                           # the underlying backend (LocalSearchBackend / Neo4jBackend / etc.)
    rerank_topk: int = 30               # how many candidates to feed to the cross-encoder
    final_n: int = 10                   # how many to return after reranking
    model_name: str = "BAAI/bge-reranker-v2-m3"

    def __post_init__(self):
        # Eagerly fetch the singleton so the first query doesn't pay startup latency.
        _get_reranker(self.model_name)

    # Expose source_filters so the runners can install per-corpus scopes.
    @property
    def source_filters(self):
        return self.base.source_filters

    @source_filters.setter
    def source_filters(self, v):
        self.base.source_filters = v

    def _rerank(self, query: str, chunks: List[Dict]) -> List[Dict]:
        if not chunks:
            return []
        ce = _get_reranker(self.model_name)
        pairs = [(query, (c.get("text") or "")[:1500]) for c in chunks]
        scores = ce.predict(pairs, batch_size=32, show_progress_bar=False)
        scored = sorted(zip(chunks, scores), key=lambda x: -float(x[1]))
        out = []
        for c, s in scored[: self.final_n]:
            c2 = dict(c)
            c2["rerank_score"] = float(s)
            c2["search_type"] = (c.get("search_type", "") + "_rerank").strip("_")
            out.append(c2)
        return out

    # ── pass-through with rerank ──
    def scoped_vector_search(self, query, agent_name, *, year=None, section=None,
                             limit: Optional[int] = None):
        topk = self.rerank_topk
        cand = self.base.scoped_vector_search(query, agent_name, year=year,
                                               section=section, limit=topk)
        return self._rerank(query, cand)[: (limit or self.final_n)]

    def scoped_fulltext_search(self, query, agent_name, *, year=None,
                               limit: Optional[int] = None):
        topk = self.rerank_topk
        cand = self.base.scoped_fulltext_search(query, agent_name, year=year, limit=topk)
        return self._rerank(query, cand)[: (limit or self.final_n)]

    def combined_scoped_search(self, query, agent_name, *, year=None,
                                section=None, limit: Optional[int] = None):
        topk = self.rerank_topk
        cand = self.base.combined_scoped_search(query, agent_name, year=year,
                                                 section=section, limit=topk)
        return self._rerank(query, cand)[: (limit or self.final_n)]

    def global_search(self, query, *, year=None, limit: Optional[int] = None):
        topk = self.rerank_topk
        cand = self.base.global_search(query, year=year, limit=topk)
        return self._rerank(query, cand)[: (limit or self.final_n)]
