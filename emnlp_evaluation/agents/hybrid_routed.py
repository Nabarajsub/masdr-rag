"""
Hybrid-Routed system: regex fast-path → LLM router → single scoped agent.

The paper's "Hybrid-Routed" configuration. This is what we expect to win the
Pareto curve against ReAct: ONE LLM call to pick a domain, then ONE retrieval,
then ONE LLM call to answer. No iterative reasoning.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..llm_providers.base import LLMProvider
from . import tool_catalog as _tc
from .tools_oss import SearchBackend, format_chunks


# Cheap regex hints, ordered by specificity. First hit wins.
# WYDOT defaults; runners override per corpus via configure_router().
_WYDOT_REGEX_RULES: List[tuple] = [
    (r"\b(standard\s+spec(?:s|ifications)?|section\s+\d{3}(?:\.\d+)*)\b", "specs_agent"),
    (r"\b(construction manual|inspection procedure)\b", "construction_agent"),
    (r"\b(materials testing|aggregate test|asphalt content|sampling method)\b", "materials_agent"),
    (r"\b(design manual|geometric design|superelevation|sight distance)\b", "design_agent"),
    (r"\b(crash|fatalit(?:y|ies)|impaired driv|traffic safety)\b", "safety_agent"),
    (r"\b(bridge|load rating|approach slab)\b", "bridge_agent"),
    (r"\b(stip|corridor study|long range plan|transportation improvement)\b", "planning_agent"),
    (r"\b(annual report|operating budget|dbe goal|financial report)\b", "admin_agent"),
]

_REGEX_RULES: List[tuple] = list(_WYDOT_REGEX_RULES)

_ROUTER_DOMAIN = "a knowledge graph of Wyoming Department of Transportation documents"


def configure_router(*, regex_rules: Optional[List[tuple]] = None,
                     domain_desc: Optional[str] = None) -> None:
    """Point the router at a different corpus. Runners that install a
    per-corpus tool catalog must call this too, otherwise the regex rules
    and router prompt keep routing against WYDOT vocabulary."""
    global _REGEX_RULES, _ROUTER_DOMAIN
    if regex_rules is not None:
        _REGEX_RULES = list(regex_rules)
    if domain_desc is not None:
        _ROUTER_DOMAIN = domain_desc


def _agent_catalog() -> Dict[str, str]:
    """agent_name -> tool description, read live from the installed tool
    catalog (runners mutate tool_catalog in place, so this must not be
    snapshotted at import time)."""
    descs = {}
    for tool, agent in _tc.TOOL_TO_AGENT.items():
        descs[agent] = _tc._TOOL_DESCRIPTIONS.get(tool, "")
    return descs


ROUTER_PROMPT = """You are a query router for {domain}.

Given a user question, choose the ONE agent whose data source is most likely to contain the answer:
{agents}

If no specific source clearly fits, answer general_agent.
Respond with ONLY the agent name, nothing else.

User question: {query}
Agent name:"""


# Prompt-control override (final_v3 E5) — set via naive_rag.set_shared_answer_prompt().
SHARED_ANSWER_PROMPT = None


def _answer_prompt() -> str:
    return SHARED_ANSWER_PROMPT or ANSWER_PROMPT


ANSWER_PROMPT = """You are the WYDOT Knowledge Graph Assistant.

Answer the user's question using ONLY the retrieved sources below. Cite as [Source 1], [Source 2], etc. Be thorough but concise. If the sources don't fully answer the question, share what you found and note the gap.

Retrieved sources:
{context}

User question: {query}
Answer:"""


@dataclass
class HybridTrace:
    answer: str = ""
    chunks: List[Dict] = field(default_factory=list)
    routed_agent: Optional[str] = None
    route_decision: str = ""  # "regex" or "llm"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    wall_time_s: float = 0.0


def regex_route(query: str) -> Optional[str]:
    for pattern, agent in _REGEX_RULES:
        if re.search(pattern, query, flags=re.IGNORECASE):
            return agent
    return None


def llm_route(query: str, llm: LLMProvider) -> tuple[str, int, int]:
    catalog = _agent_catalog()
    agent_lines = "\n".join(
        f"- {a}: {d}" if d else f"- {a}" for a, d in sorted(catalog.items()))
    prompt = ROUTER_PROMPT.format(domain=_ROUTER_DOMAIN, agents=agent_lines, query=query)
    res = llm.generate(
        [{"role": "user", "content": prompt}],
        max_new_tokens=32, temperature=0.0,
    )
    # Case-insensitive match against the catalog. The previous version lower-cased
    # the model output but compared it with the catalog names as-is, so any scope
    # with capitals or spaces (every WYDOT document_series, e.g. "Standard
    # Specs_agent") could never be selected and the router silently fell back to
    # general_agent (found in RQ2 verification, 2026-09-28).
    by_lower = {a.lower(): a for a in catalog}
    text = (res.text or "").strip()
    first = text.splitlines()[0].strip().strip(":,.`'\"").lower() if text else ""
    if first in by_lower:
        pick = by_lower[first]
    else:
        hits = [a for a in by_lower if a in text.lower()]
        pick = by_lower[max(hits, key=len)] if hits else "general_agent"
    return pick, res.prompt_tokens, res.completion_tokens


def run_hybrid_routed(
    query: str, *,
    llm: LLMProvider, backend: SearchBackend,
    router_llm: Optional[LLMProvider] = None,
    use_regex_fastpath: bool = True,
) -> HybridTrace:
    """Regex → LLM route → scoped retrieve → answer."""
    router_llm = router_llm or llm
    trace = HybridTrace()
    t0 = time.time()

    agent = regex_route(query) if use_regex_fastpath else None
    if agent:
        trace.route_decision = "regex"
    else:
        agent, pt, ct = llm_route(query, router_llm)
        trace.prompt_tokens += pt
        trace.completion_tokens += ct
        trace.llm_calls += 1
        trace.route_decision = "llm"
    trace.routed_agent = agent

    if agent == "general_agent":
        chunks = backend.global_search(query)
    else:
        chunks = backend.combined_scoped_search(query, agent)
    trace.chunks = chunks

    answer_prompt = _answer_prompt().format(context=format_chunks(chunks), query=query)
    res = llm.generate(
        [{"role": "user", "content": answer_prompt}],
        max_new_tokens=1024,
    )
    trace.answer = res.text
    trace.prompt_tokens += res.prompt_tokens
    trace.completion_tokens += res.completion_tokens
    trace.llm_calls += 1
    trace.wall_time_s = time.time() - t0
    return trace


def run_regex_scoped(query: str, *, llm: LLMProvider, backend: SearchBackend) -> HybridTrace:
    """Regex-only routing (no LLM router). Falls back to general_agent."""
    trace = HybridTrace()
    t0 = time.time()
    agent = regex_route(query) or "general_agent"
    trace.routed_agent = agent
    trace.route_decision = "regex"
    if agent == "general_agent":
        chunks = backend.global_search(query)
    else:
        chunks = backend.combined_scoped_search(query, agent)
    trace.chunks = chunks

    res = llm.generate(
        [{"role": "user", "content": _answer_prompt().format(
            context=format_chunks(chunks), query=query)}],
        max_new_tokens=1024,
    )
    trace.answer = res.text
    trace.prompt_tokens += res.prompt_tokens
    trace.completion_tokens += res.completion_tokens
    trace.llm_calls += 1
    trace.wall_time_s = time.time() - t0
    return trace


class CentroidRouter:
    """Training-free soft router: cosine(query emb, per-scope centroid).

    Centroids are mean chunk embeddings per scope (L2-normalized), so this
    needs no gold routing labels and works on any corpus — including ones
    where regex/LLM routing has no signal.
    """

    def __init__(self, centroids: Dict[str, "np.ndarray"]):
        import numpy as np
        self._np = np
        self.scopes = sorted(centroids)
        self._mat = np.stack([centroids[s] for s in self.scopes])
        self._mat /= (np.linalg.norm(self._mat, axis=1, keepdims=True) + 1e-12)

    @classmethod
    def from_corpus(cls, meta, embeddings, scope_field: str) -> "CentroidRouter":
        import numpy as np
        labels = meta[scope_field].astype(str).to_numpy()
        cents = {}
        for s in sorted(set(labels)):
            cents[s] = np.asarray(embeddings[labels == s], dtype=np.float64).mean(0)
        return cls(cents)

    def top_m(self, query_emb, m: int = 2) -> List[tuple]:
        """[(scope, cosine), ...] best-first."""
        np = self._np
        q = np.asarray(query_emb, dtype=np.float64)
        q /= (np.linalg.norm(q) + 1e-12)
        sims = self._mat @ q
        order = np.argsort(-sims)[:m]
        return [(self.scopes[i], float(sims[i])) for i in order]


def _rrf_fuse(lists: List[List[Dict]], *, k: int = 60, limit: int = 15,
              weights: Optional[List[float]] = None) -> List[Dict]:
    """Weighted reciprocal-rank fusion across ranked chunk lists, dedup by id."""
    weights = weights or [1.0] * len(lists)
    scores: Dict[str, float] = {}
    best: Dict[str, Dict] = {}
    for w, lst in zip(weights, lists):
        for rank, ch in enumerate(lst):
            cid = ch["id"]
            scores[cid] = scores.get(cid, 0.0) + w / (k + rank + 1)
            if cid not in best:
                best[cid] = ch
    ranked = sorted(scores, key=scores.get, reverse=True)[:limit]
    out = []
    for cid in ranked:
        ch = dict(best[cid])
        ch["score"] = scores[cid]
        out.append(ch)
    return out


def run_soft_scoped(query: str, *, llm: LLMProvider, backend: SearchBackend,
                    centroid_router: CentroidRouter, m: int = 2,
                    rrf_k: int = 60, limit: int = 15,
                    global_weight: float = 1.0) -> HybridTrace:
    """SOFT-SCOPED: top-m centroid-routed scoped retrievals + global retrieval,
    RRF-fused, single synthesis call.

    The global list always participates in the fusion, so with no routing
    signal the system degrades toward monolithic instead of hard-scoping
    into the wrong partition."""
    trace = HybridTrace()
    t0 = time.time()
    q_emb = backend.embedder.embed_query(query)
    picks = centroid_router.top_m(q_emb, m=m)
    trace.routed_agent = ",".join(s for s, _ in picks)
    trace.route_decision = "centroid_soft"

    lists = [backend.combined_scoped_search(query, f"{s}_agent", limit=limit)
             for s, _ in picks]
    lists.append(backend.global_search(query, limit=limit))
    weights = [1.0] * len(picks) + [global_weight]
    trace.chunks = _rrf_fuse(lists, k=rrf_k, limit=limit, weights=weights)

    res = llm.generate(
        [{"role": "user", "content": _answer_prompt().format(
            context=format_chunks(trace.chunks), query=query)}],
        max_new_tokens=1024,
    )
    trace.answer = res.text
    trace.prompt_tokens += res.prompt_tokens
    trace.completion_tokens += res.completion_tokens
    trace.llm_calls += 1
    trace.wall_time_s = time.time() - t0
    return trace


def run_oracle_scoped(query: str, agent: str, *, llm: LLMProvider,
                      backend: SearchBackend) -> HybridTrace:
    """ORACLE hard scoping: retrieve inside the query's GOLD scope (given, not
    predicted). Upper bound for hard partitioning — separates "the router picked
    the wrong scope" from "restricting to one scope hurts even when it is right"."""
    trace = HybridTrace()
    t0 = time.time()
    trace.routed_agent = agent
    trace.route_decision = "oracle"
    if agent == "general_agent":
        chunks = backend.global_search(query)
    else:
        chunks = backend.combined_scoped_search(query, agent)
    trace.chunks = chunks
    res = llm.generate(
        [{"role": "user", "content": _answer_prompt().format(
            context=format_chunks(chunks), query=query)}],
        max_new_tokens=1024,
    )
    trace.answer = res.text
    trace.prompt_tokens += res.prompt_tokens
    trace.completion_tokens += res.completion_tokens
    trace.llm_calls += 1
    trace.wall_time_s = time.time() - t0
    return trace


def run_r2_routed(query: str, *, llm: LLMProvider, backend: SearchBackend,
                  router=None) -> HybridTrace:
    """R2 (BGE-M3 + LogReg) router → scoped retrieve → answer. No LLM
    router call, no regex fallback — this is the end-to-end Hybrid-Routed
    variant with the trained classifier head plugged in. Pass `router` to
    use a non-default artifact/label space (e.g. discovered scopes)."""
    from .r2_router import R2Router
    trace = HybridTrace()
    t0 = time.time()
    agent = (router or R2Router.get()).route_top1(query)
    trace.routed_agent = agent
    trace.route_decision = "r2"
    if agent == "general_agent":
        chunks = backend.global_search(query)
    else:
        chunks = backend.combined_scoped_search(query, agent)
    trace.chunks = chunks

    res = llm.generate(
        [{"role": "user", "content": _answer_prompt().format(
            context=format_chunks(chunks), query=query)}],
        max_new_tokens=1024,
    )
    trace.answer = res.text
    trace.prompt_tokens += res.prompt_tokens
    trace.completion_tokens += res.completion_tokens
    trace.llm_calls += 1
    trace.wall_time_s = time.time() - t0
    return trace

def run_composed(query: str, *, llm: LLMProvider, backend: SearchBackend,
                 centroid_router: "CentroidRouter", n_scopes: int, total: int = 15) -> HybridTrace:
    """RQ5 control: ONE synthesis call over a fixed-size context (`total` chunks)
    drawn evenly from the query's top-`n_scopes` centroid scopes, interleaved
    round-robin. Varies only how many distinct sources the generator must
    reconcile; chunk count, prompt and call count are held fixed."""
    trace = HybridTrace()
    t0 = time.time()
    q_emb = backend.embedder.embed_query(query)
    picks = centroid_router.top_m(q_emb, m=n_scopes)
    trace.routed_agent = ",".join(s for s, _ in picks)
    trace.route_decision = f"composed_{n_scopes}"
    per = [total // len(picks) + (1 if i < total % len(picks) else 0) for i in range(len(picks))]
    lists = [backend.combined_scoped_search(query, f"{s}_agent", limit=k)
             for (s, _), k in zip(picks, per)]
    chunks, seen = [], set()
    for i in range(max(per)):
        for lst in lists:
            if i < len(lst) and lst[i]["id"] not in seen:
                seen.add(lst[i]["id"]); chunks.append(lst[i])
    trace.chunks = chunks[:total]
    res = llm.generate(
        [{"role": "user", "content": _answer_prompt().format(
            context=format_chunks(trace.chunks), query=query)}],
        max_new_tokens=1024,
    )
    trace.answer = res.text
    trace.prompt_tokens += res.prompt_tokens
    trace.completion_tokens += res.completion_tokens
    trace.llm_calls += 1
    trace.wall_time_s = time.time() - t0
    return trace
