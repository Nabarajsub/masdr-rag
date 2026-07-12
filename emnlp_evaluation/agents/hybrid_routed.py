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
    agent_names = set(catalog)
    pick = res.text.strip().split()[0].lower().strip(":,.") if res.text else "general_agent"
    if pick not in agent_names:
        # try lenient matching
        for a in agent_names:
            if a in res.text.lower():
                pick = a; break
        else:
            pick = "general_agent"
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

    answer_prompt = ANSWER_PROMPT.format(context=format_chunks(chunks), query=query)
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
        [{"role": "user", "content": ANSWER_PROMPT.format(
            context=format_chunks(chunks), query=query)}],
        max_new_tokens=1024,
    )
    trace.answer = res.text
    trace.prompt_tokens += res.prompt_tokens
    trace.completion_tokens += res.completion_tokens
    trace.llm_calls += 1
    trace.wall_time_s = time.time() - t0
    return trace


def run_r2_routed(query: str, *, llm: LLMProvider, backend: SearchBackend) -> HybridTrace:
    """R2 (BGE-M3 + LogReg) router → scoped retrieve → answer. No LLM
    router call, no regex fallback — this is the end-to-end Hybrid-Routed
    variant with the trained classifier head plugged in."""
    from .r2_router import R2Router
    trace = HybridTrace()
    t0 = time.time()
    agent = R2Router.get().route_top1(query)
    trace.routed_agent = agent
    trace.route_decision = "r2"
    if agent == "general_agent":
        chunks = backend.global_search(query)
    else:
        chunks = backend.combined_scoped_search(query, agent)
    trace.chunks = chunks

    res = llm.generate(
        [{"role": "user", "content": ANSWER_PROMPT.format(
            context=format_chunks(chunks), query=query)}],
        max_new_tokens=1024,
    )
    trace.answer = res.text
    trace.prompt_tokens += res.prompt_tokens
    trace.completion_tokens += res.completion_tokens
    trace.llm_calls += 1
    trace.wall_time_s = time.time() - t0
    return trace
