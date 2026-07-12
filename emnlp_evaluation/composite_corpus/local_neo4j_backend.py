"""LocalNeo4jBackend — same API as LocalSearchBackend, backed by local Neo4j.

The existing runners (run_composite / run_multihop / run_generic_corpus)
talk to a backend via these methods:
    combined_scoped_search(query, agent_name, *, year=None, section=None, limit)
    scoped_vector_search(query, agent_name, *, year=None, section=None, limit)
    scoped_fulltext_search(query, agent_name, *, year=None, limit)
    global_search(query, *, year=None, limit)

We implement them against the local Neo4j daemon installed at
bolt://t502:24389 by emnlp_evaluation/slurm/run_local_neo4j.sbatch.

Per-corpus isolation: each dataset writes nodes with a per-corpus label
(e.g. :MultihopDoc, :MultihopChunk) so they coexist in one database.
Vector indexes are also corpus-scoped (multihop_chunk_vec, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from neo4j import GraphDatabase
from ..configs import oss_config as cfg
from ..embeddings.base import Embedder


@dataclass
class LocalNeo4jBackend:
    embedder: Embedder
    corpus: str                            # e.g. "multihop", "financebench", "wydot_faiss"
    scope_field: str = "source_type"       # node property holding scope label
    source_filters: Dict[str, List[str]] = field(default_factory=dict)  # agent -> scopes
    default_vector_k: int = cfg.DEFAULT_VECTOR_K
    default_limit: int = cfg.DEFAULT_SCOPED_LIMIT
    _driver = None

    @property
    def chunk_label(self) -> str:
        return f"{self.corpus.capitalize()}Chunk"

    @property
    def vector_index(self) -> str:
        return f"{self.corpus}_chunk_vec"

    @property
    def fulltext_index(self) -> str:
        return f"{self.corpus}_chunk_ft"

    # ── connection ────────────────────────────────────────────────────────

    def driver(self):
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                cfg.NEO4J_LOCAL_URI,
                auth=(cfg.NEO4J_LOCAL_USERNAME, cfg.NEO4J_LOCAL_PASSWORD),
            )
        return self._driver

    def _session(self):
        kw = {}
        if cfg.NEO4J_LOCAL_DATABASE:
            kw["database"] = cfg.NEO4J_LOCAL_DATABASE
        return self.driver().session(**kw)

    # ── search API mirroring LocalSearchBackend ───────────────────────────

    def _row(self, n: Dict[str, Any], score: float, kind: str) -> Dict[str, Any]:
        return {
            "id": str(n.get("id", "")),
            "text": str(n.get("text", "")),
            "source": str(n.get(self.scope_field, "")),
            "title": str(n.get("title", "")),
            "year": n.get("year"),
            "section": str(n.get("section", "")),
            "score": float(score),
            "search_type": kind,
        }

    def scoped_vector_search(self, query: str, agent_name: str, *,
                             year=None, section=None,
                             limit: Optional[int] = None) -> List[Dict]:
        limit = limit or self.default_limit
        emb = self.embedder.embed_query(query)
        allowed = self.source_filters.get(agent_name, [])
        cypher_filter = ""
        params = {"emb": emb, "k": self.default_vector_k * 3, "limit": limit}
        if allowed:
            cypher_filter = f" AND n.{self.scope_field} IN $allowed"
            params["allowed"] = allowed
        cypher = (
            f"CALL db.index.vector.queryNodes('{self.vector_index}', $k, $emb) "
            f"YIELD node AS n, score "
            f"WHERE n:{self.chunk_label}{cypher_filter} "
            f"RETURN n {{ .id, .text, .{self.scope_field}, .title, .year, .section }} AS n, score "
            f"ORDER BY score DESC LIMIT $limit"
        )
        with self._session() as s:
            return [self._row(rec["n"], rec["score"], "neo4j_vector")
                    for rec in s.run(cypher, **params)]

    def scoped_fulltext_search(self, query: str, agent_name: str, *,
                               year=None, limit: Optional[int] = None) -> List[Dict]:
        limit = limit or cfg.DEFAULT_FULLTEXT_LIMIT
        allowed = self.source_filters.get(agent_name, [])
        # Sanitise the query: drop chars that break Lucene parsers.
        safe = "".join(c if c.isalnum() or c.isspace() else " " for c in query).strip()
        if not safe:
            return []
        try:
            params = {"q": safe, "limit": limit * 3}
            cypher_filter = ""
            if allowed:
                cypher_filter = f" AND node.{self.scope_field} IN $allowed"
                params["allowed"] = allowed
            cypher = (
                f"CALL db.index.fulltext.queryNodes('{self.fulltext_index}', $q) "
                f"YIELD node, score "
                f"WHERE node:{self.chunk_label}{cypher_filter} "
                f"RETURN node {{ .id, .text, .{self.scope_field}, .title, .year, .section }} AS n, score "
                f"LIMIT $limit"
            )
            params["limit"] = limit
            with self._session() as s:
                return [self._row(rec["n"], rec["score"], "neo4j_fulltext")
                        for rec in s.run(cypher, **params)]
        except Exception:
            return []

    def combined_scoped_search(self, query: str, agent_name: str, *,
                                year=None, section=None,
                                limit: Optional[int] = None) -> List[Dict]:
        limit = limit or self.default_limit
        vec = self.scoped_vector_search(query, agent_name, limit=limit)
        ft = self.scoped_fulltext_search(query, agent_name, limit=max(5, limit // 2))
        seen = {r["id"] for r in vec}
        out = list(vec)
        for r in ft:
            if r["id"] not in seen:
                seen.add(r["id"]); out.append(r)
        return out[:limit]

    def global_search(self, query: str, *, year=None,
                      limit: Optional[int] = None) -> List[Dict]:
        limit = limit or self.default_limit
        emb = self.embedder.embed_query(query)
        cypher = (
            f"CALL db.index.vector.queryNodes('{self.vector_index}', $k, $emb) "
            f"YIELD node AS n, score "
            f"WHERE n:{self.chunk_label} "
            f"RETURN n {{ .id, .text, .{self.scope_field}, .title, .year, .section }} AS n, score "
            f"ORDER BY score DESC LIMIT $limit"
        )
        with self._session() as s:
            return [self._row(rec["n"], rec["score"], "neo4j_vector_global")
                    for rec in s.run(cypher, emb=emb, k=self.default_vector_k * 3, limit=limit)]


def load_neo4j_backend(corpus: str, embedder: Embedder, *,
                       scope_field: str = "source_type",
                       source_filters: Optional[Dict[str, List[str]]] = None) -> LocalNeo4jBackend:
    return LocalNeo4jBackend(
        embedder=embedder, corpus=corpus,
        scope_field=scope_field,
        source_filters=source_filters or {},
    )
