"""
Embed-agnostic, index-agnostic scoped search tools.

Mirrors `agentic_solution/tools.py` but the embedder and the vector index name
are injected, so the same tool code can run against the Gemini index or the
BGE-M3 index without changes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from neo4j import GraphDatabase

from ..configs import oss_config as cfg
from ..embeddings.base import Embedder

# Reuse the production filter dict so we never drift.
from agentic_solution.config import AGENT_SERIES_FILTERS


_STOPWORDS = {
    "what", "which", "where", "when", "how", "who", "does", "did", "the",
    "is", "are", "was", "were", "been", "being", "have", "has", "had",
    "for", "and", "but", "not", "this", "that", "with", "from", "about",
    "between", "into", "through", "during", "before", "after", "above",
    "below", "than", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "only", "same", "also", "tell", "show",
    "describe", "explain", "compare", "list", "give", "find", "need",
    "requirements", "required", "requirement", "according", "specific",
    "used", "using", "must", "shall", "should", "would",
}


def _keywords(query: str) -> List[str]:
    words = re.sub(r"[^\w\s]", " ", query).split()
    return [w for w in words if len(w) > 2 and w.lower() not in _STOPWORDS][:8]


def _where_clause(agent_name: str) -> str:
    filters = AGENT_SERIES_FILTERS.get(agent_name, [])
    if not filters:
        return "TRUE"
    return " OR ".join(f"({f})" for f in filters)


@dataclass
class SearchBackend:
    """Bundle of (embedder, vector index, fulltext index) used by all tools."""
    embedder: Embedder
    vector_index: str
    fulltext_index: str = cfg.NEO4J_FULLTEXT_INDEX

    _driver = None

    def driver(self):
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                cfg.NEO4J_URI, auth=(cfg.NEO4J_USERNAME, cfg.NEO4J_PASSWORD),
            )
        return self._driver

    def _session(self):
        # Force READ access mode: any accidental CREATE/MERGE/SET/DELETE against
        # production Neo4j raises ClientError at the driver level. Reads are the
        # only operation this harness performs on production (per user policy);
        # writes go to the local sandbox via NEO4J_LOCAL_URI.
        from neo4j import READ_ACCESS
        kw = {"default_access_mode": READ_ACCESS}
        if cfg.NEO4J_DATABASE:
            kw["database"] = cfg.NEO4J_DATABASE
        return self.driver().session(**kw)

    # ── scoped vector search ────────────────────────────────────────────────

    def scoped_vector_search(
        self, query: str, agent_name: str, *,
        year: Optional[int] = None, section: Optional[str] = None,
        limit: int = cfg.DEFAULT_SCOPED_LIMIT,
    ) -> List[Dict]:
        emb = self.embedder.embed_query(query)
        where = _where_clause(agent_name)
        extra, params = [], {
            "k": cfg.DEFAULT_VECTOR_K * 3,
            "emb": emb,
            "limit": limit,
        }
        if year:
            extra.append("d.year = $year"); params["year"] = year
        if section:
            extra.append("s.name CONTAINS $sec"); params["sec"] = section
        extra_clause = (" AND " + " AND ".join(extra)) if extra else ""

        cypher = f"""
        CALL db.index.vector.queryNodes('{self.vector_index}', $k, $emb)
        YIELD node, score
        MATCH (d:Document)-[:HAS_SECTION]->(s:Section)-[:HAS_CHUNK]->(node)
        WHERE ({where}){extra_clause}
        RETURN node.id AS id, node.text AS text, d.source AS source,
               d.display_title AS title, d.year AS year,
               s.name AS section, node.page AS page, score
        ORDER BY score DESC LIMIT $limit
        """
        return self._run(cypher, params, "scoped_vector")

    # ── scoped fulltext search ──────────────────────────────────────────────

    def scoped_fulltext_search(
        self, query: str, agent_name: str, *,
        year: Optional[int] = None, limit: int = cfg.DEFAULT_FULLTEXT_LIMIT,
    ) -> List[Dict]:
        kws = _keywords(query)
        if not kws:
            return []
        params = {"ftq": " ".join(kws), "limit": limit}
        year_filter = ""
        if year:
            year_filter = " AND d.year = $year"; params["year"] = year
        where = _where_clause(agent_name)
        cypher = f"""
        CALL db.index.fulltext.queryNodes('{self.fulltext_index}', $ftq)
        YIELD node, score
        MATCH (d:Document)-[:HAS_SECTION]->(s:Section)-[:HAS_CHUNK]->(node)
        WHERE ({where}){year_filter}
        RETURN node.id AS id, node.text AS text, d.source AS source,
               d.display_title AS title, d.year AS year,
               s.name AS section, node.page AS page, score
        ORDER BY score DESC LIMIT $limit
        """
        return self._run(cypher, params, "scoped_fulltext")

    # ── global (unscoped) search ────────────────────────────────────────────

    def global_search(
        self, query: str, *, year: Optional[int] = None,
        limit: int = cfg.DEFAULT_SCOPED_LIMIT,
    ) -> List[Dict]:
        emb = self.embedder.embed_query(query)
        params = {"k": cfg.DEFAULT_VECTOR_K * 2, "emb": emb, "limit": limit}
        year_filter = ""
        if year:
            year_filter = " AND d.year = $year"; params["year"] = year
        cypher = f"""
        CALL db.index.vector.queryNodes('{self.vector_index}', $k, $emb)
        YIELD node, score
        MATCH (d:Document)-[:HAS_SECTION]->(s:Section)-[:HAS_CHUNK]->(node)
        WHERE TRUE{year_filter}
        RETURN node.id AS id, node.text AS text, d.source AS source,
               d.display_title AS title, d.year AS year,
               s.name AS section, node.page AS page, score
        ORDER BY score DESC LIMIT $limit
        """
        return self._run(cypher, params, "global_vector")

    def combined_scoped_search(
        self, query: str, agent_name: str, *,
        year: Optional[int] = None, section: Optional[str] = None,
        limit: int = cfg.DEFAULT_SCOPED_LIMIT,
    ) -> List[Dict]:
        """Vector + fulltext merged, deduplicated by chunk id."""
        vec = self.scoped_vector_search(query, agent_name, year=year, section=section, limit=limit)
        ft = self.scoped_fulltext_search(query, agent_name, year=year, limit=max(5, limit // 2))
        seen = {r["id"] for r in vec}
        out = list(vec)
        for r in ft:
            if r["id"] not in seen:
                seen.add(r["id"]); out.append(r)
        return out[:limit]

    # ── internal ─────────────────────────────────────────────────────────────

    def _run(self, cypher: str, params: dict, search_type: str) -> List[Dict]:
        out: List[Dict] = []
        seen = set()
        try:
            with self._session() as sess:
                for rec in sess.run(cypher, **params):
                    cid = rec["id"]
                    if cid in seen:
                        continue
                    seen.add(cid)
                    out.append({
                        "id": cid,
                        "text": rec["text"],
                        "source": rec["source"],
                        "title": rec["title"],
                        "year": rec["year"],
                        "section": rec["section"],
                        "page": rec["page"],
                        "score": rec["score"],
                        "search_type": search_type,
                    })
        except Exception as e:
            print(f"    [{search_type}] error: {e}")
        return out


def format_chunks(chunks: List[Dict]) -> str:
    if not chunks:
        return "No results found."
    parts = []
    for i, c in enumerate(chunks, 1):
        head = f"[SOURCE {i}: {c.get('title','Unknown')}, {c.get('section','')}, Year: {c.get('year','?')}]"
        parts.append(f"{head}\n{c.get('text','')}\n")
    return "\n---\n".join(parts)
