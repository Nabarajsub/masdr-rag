"""
SearchBackend specialised for the MultiHop-RAG subgraph.

Mirrors `agents.tools_oss.SearchBackend` so the same orchestrator / hybrid /
ReAct code can run unmodified — we just inject a different backend.

The scoping filter is `source = <publisher>` instead of document_series.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from neo4j import GraphDatabase

from ...configs import oss_config as cfg
from ...embeddings.base import Embedder
from .ingest import MULTIHOP_INDEX, MULTIHOP_FULLTEXT_INDEX
from .source_map import build_source_filters


def _where(filters: Dict[str, List[str]], agent: str) -> str:
    return " OR ".join(f"({f})" for f in filters.get(agent, [])) or "TRUE"


@dataclass
class MultihopSearchBackend:
    embedder: Embedder
    source_filters: Dict[str, List[str]]
    vector_index: str = MULTIHOP_INDEX
    fulltext_index: str = MULTIHOP_FULLTEXT_INDEX
    _driver = None

    def driver(self):
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                cfg.NEO4J_URI, auth=(cfg.NEO4J_USERNAME, cfg.NEO4J_PASSWORD),
            )
        return self._driver

    def _session(self):
        kw = {"database": cfg.NEO4J_DATABASE} if cfg.NEO4J_DATABASE else {}
        return self.driver().session(**kw)

    # The orchestrator/ReAct code calls combined_scoped_search / global_search.
    def combined_scoped_search(self, query, agent_name, *, year=None, section=None,
                                limit: int = cfg.DEFAULT_SCOPED_LIMIT) -> List[Dict]:
        vec = self.scoped_vector_search(query, agent_name, limit=limit)
        ft = self.scoped_fulltext_search(query, agent_name, limit=max(5, limit // 2))
        seen = {r["id"] for r in vec}; out = list(vec)
        for r in ft:
            if r["id"] not in seen:
                seen.add(r["id"]); out.append(r)
        return out[:limit]

    def scoped_vector_search(self, query, agent_name, *, limit=cfg.DEFAULT_SCOPED_LIMIT,
                             year=None, section=None) -> List[Dict]:
        emb = self.embedder.embed_query(query)
        where = _where(self.source_filters, agent_name)
        cypher = f"""
        CALL db.index.vector.queryNodes('{self.vector_index}', $k, $emb)
        YIELD node, score
        MATCH (m:MultihopDoc)-[:HAS_CHUNK]->(node)
        WHERE ({where})
        RETURN node.id AS id, node.text AS text, m.source AS source,
               m.title AS title, m.published_at AS year, '' AS section,
               node.idx AS page, score
        ORDER BY score DESC LIMIT $limit
        """
        return self._run(cypher, {"k": cfg.DEFAULT_VECTOR_K * 3, "emb": emb, "limit": limit},
                         "mh_scoped_vector")

    def scoped_fulltext_search(self, query, agent_name, *, year=None, limit=10) -> List[Dict]:
        where = _where(self.source_filters, agent_name)
        cypher = f"""
        CALL db.index.fulltext.queryNodes('{self.fulltext_index}', $ftq)
        YIELD node, score
        MATCH (m:MultihopDoc)-[:HAS_CHUNK]->(node)
        WHERE ({where})
        RETURN node.id AS id, node.text AS text, m.source AS source,
               m.title AS title, m.published_at AS year, '' AS section,
               node.idx AS page, score
        ORDER BY score DESC LIMIT $limit
        """
        return self._run(cypher, {"ftq": query, "limit": limit}, "mh_scoped_fulltext")

    def global_search(self, query, *, year=None, limit=cfg.DEFAULT_SCOPED_LIMIT) -> List[Dict]:
        emb = self.embedder.embed_query(query)
        cypher = f"""
        CALL db.index.vector.queryNodes('{self.vector_index}', $k, $emb)
        YIELD node, score
        MATCH (m:MultihopDoc)-[:HAS_CHUNK]->(node)
        RETURN node.id AS id, node.text AS text, m.source AS source,
               m.title AS title, m.published_at AS year, '' AS section,
               node.idx AS page, score
        ORDER BY score DESC LIMIT $limit
        """
        return self._run(cypher, {"k": cfg.DEFAULT_VECTOR_K * 2, "emb": emb, "limit": limit},
                         "mh_global_vector")

    def _run(self, cypher, params, kind) -> List[Dict]:
        out, seen = [], set()
        try:
            with self._session() as s:
                for rec in s.run(cypher, **params):
                    cid = rec["id"]
                    if cid in seen: continue
                    seen.add(cid)
                    out.append({
                        "id": cid, "text": rec["text"],
                        "source": rec["source"], "title": rec["title"],
                        "year": rec["year"], "section": rec["section"],
                        "page": rec["page"], "score": rec["score"],
                        "search_type": kind,
                    })
        except Exception as e:
            print(f"    [{kind}] {e}")
        return out


def make_multihop_backend(embedder: Embedder) -> MultihopSearchBackend:
    from .source_map import discover_sources
    sources = discover_sources()
    return MultihopSearchBackend(
        embedder=embedder,
        source_filters=build_source_filters(sources),
    )
