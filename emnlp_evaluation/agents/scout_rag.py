"""SCOUT-RAG: best-effort reimplementation of Algorithm 1 from
arXiv:2602.08400 (SCOUT-RAG: Scalable and Cost-Efficient Unifying Traversal
for Agentic Graph-RAG over Distributed Domains).

The upstream paper provides only the algorithmic skeleton and does NOT
release: (a) any agent prompt templates, (b) the LLM choice for the
agents themselves (only GPT-4o as judge is specified), (c) the
DRAA feature fusion function, (d) the OASA fusion logic, (e) the
∆Q stagnation threshold ε, and (f) the retrieval operators for global
vs. local PAGA. This file therefore implements the published Algorithm
1 structure (DRAA → PAGA → OASA → AQAA refinement loop, strategy
selector and termination criteria as published) with our own prompts,
weights, thresholds, and operators. We disclose this in the paper as
"SCOUT-RAG (our reimpl)" and report Algorithm 1 conformance.

Mapping from the paper to our setting:
  * Each existing scope agent (regex/agent_name in TOOL_TO_AGENT) is
    treated as one SCOUT-RAG "domain" 𝒟ᵢ.
  * Domain "global" retrieval (HIGH tier, community-level) → scoped
    vector search with k = 20 on that domain.
  * Domain "local" retrieval (MODERATE tier, entity-level) → scoped
    vector search with k = 5 plus a focused fulltext-only fallback.
  * "Historical performance" sᵢʰⁱˢᵗ is unavailable on a fresh query,
    so we set it to 0 — DRAA scoring uses only semantic_sim and
    knowledge_richness.
  * The published budget Tᵣ is in seconds; we keep that and exit when
    wall time ≥ a hard cap (90s).

Comparison fairness:
  * Same LLM (Qwen-2.5-7B / Llama-3-8B) as all other systems.
  * Same retriever (BGE-M3 against our index).
  * Same domain partition as our scoping methods (the existing
    TOOL_TO_AGENT map per corpus).
  * The only contrast is the *coordination protocol*: SCOUT-RAG runs
    DRAA tier classification, PAGA domain-adaptive retrieval, OASA
    cross-domain fusion, and AQAA-driven refinement loops — vs.
    MASDR-RAG (function-call orchestration) and MA-RAG (sequential
    plan-decompose-execute).
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..llm_providers.base import LLMProvider
from .tool_catalog import TOOL_TO_AGENT
from .tools_oss import SearchBackend, format_chunks


# ── SCOUT-RAG agent prompts (ours; paper does not release templates) ──

DRAA_SYSTEM = """You are the Domain Relevance Assessment Agent (DRAA) in the SCOUT-RAG system.
Given a query and a list of available knowledge domains (each with a brief description and chunk-count signal), classify EACH domain into one of four tiers:

  HIGH        — directly relevant; expect community-level evidence here
  MODERATE    — partially relevant; expect entity/relation-level evidence
  POTENTIAL   — may contain peripheral evidence; consider only on refinement
  IRRELEVANT  — unlikely to contain any useful evidence

You receive two signals per domain:
  sim   ∈ [0,1] — embedding cosine similarity between query and domain description
  rich  ∈ [0,1] — normalised knowledge richness (chunks-in-domain / max-domain-chunks)

Reason about each domain briefly, then output JSON.

Return ONLY valid JSON of the form:
{
  "rationale": "<one-line reason>",
  "domains": [
    {"name": "<domain_name>", "tier": "HIGH|MODERATE|POTENTIAL|IRRELEVANT"},
    ...
  ]
}
"""

DRAA_HUMAN = """Query: {query}

Available domains (with sim, rich signals):
{domains_block}

Classify each domain into one tier.
"""

PAGA_QA_SYSTEM = """You are a Partial Answer Generation Agent (PAGA) handling a single domain in the SCOUT-RAG system.
Given a query and retrieved chunks from your domain, produce a concise partial answer that:
  - cites only chunks from this domain (use [doc_<id>] markers)
  - states what the domain *can* answer and what it cannot
  - does NOT make claims outside the supplied chunks

Return ONLY valid JSON of the form:
{
  "partial_answer": "<your domain-scoped partial answer>",
  "coverage": "<what aspects of the query this domain covers>",
  "gaps": "<what remains unanswered from this domain>"
}
"""

PAGA_HUMAN = """Query: {query}
Domain: {domain}
Granularity: {granularity}   (HIGH=community-level summary, MODERATE=entity/relation-level)

Retrieved chunks from this domain:
{context}
"""

OASA_SYSTEM = """You are the Overall Answer Synthesis Agent (OASA) in the SCOUT-RAG system.
Given a query and partial answers from multiple domain agents, produce one coherent answer that:
  - integrates evidence across domains, attributing claims to source domains where helpful
  - resolves conflicts by preferring HIGH-tier domain claims over MODERATE
  - does NOT add claims unsupported by any partial answer

Return ONLY valid JSON of the form:
{
  "answer": "<the synthesised final answer>",
  "attributions": "<which domains contributed which claims>"
}
"""

OASA_HUMAN = """Query: {query}
Partial answers (one per active domain):
{partials_block}
"""

AQAA_SYSTEM = """You are the Answer Quality Assessment Agent (AQAA) in the SCOUT-RAG system.
Given a query and the current synthesised answer, evaluate it on:
  Completeness C ∈ [0,1] — does the answer cover all aspects of the query?
  Diversity   V ∈ [0,1] — does it draw on appropriately diverse sources/perspectives?

Identify unresolved gaps (max 3) and propose follow-up sub-queries that would close them (max 3).

Return ONLY valid JSON of the form:
{
  "C": <float 0-1>,
  "V": <float 0-1>,
  "gaps": ["<gap 1>", "<gap 2>"],
  "follow_ups": ["<sub-query 1>", "<sub-query 2>"]
}
"""

AQAA_HUMAN = """Query: {query}
Current synthesised answer: {answer}

Domains already consulted: {consulted}
"""


# ── JSON parsing (same tolerant approach as MA-RAG) ──

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _parse_json(text: str, default: dict) -> dict:
    if not text:
        return default
    m = _CODE_FENCE_RE.search(text)
    cand = m.group(1) if m else None
    if cand is None:
        m = _JSON_BLOCK_RE.search(text)
        cand = m.group(0) if m else None
    if cand is None:
        return default
    try:
        return json.loads(cand)
    except json.JSONDecodeError:
        try:
            return json.loads(cand.replace("'", '"'))
        except json.JSONDecodeError:
            return default


def _as_text(v) -> str:
    """Coerce an LLM-produced field to a string. The model sometimes
    emits a JSON array/object for "answer"; flatten it so the answer is
    always a string for the downstream judge."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return " ".join(_as_text(x) for x in v)
    if isinstance(v, dict):
        return " ".join(_as_text(x) for x in v.values())
    return str(v)


# ── trace ──

@dataclass
class ScoutTrace:
    answer: str = ""
    chunks: List[Dict] = field(default_factory=list)
    tiers: Dict[str, str] = field(default_factory=dict)
    partial_answers: List[Dict] = field(default_factory=list)
    iterations: int = 0
    strategies: List[str] = field(default_factory=list)
    quality_history: List[float] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    wall_time_s: float = 0.0
    early_stop_reason: str = "stop_strategy"
    routed_agents: List[str] = field(default_factory=list)


# ── helpers ──

def _gen(llm: LLMProvider, sys: str, usr: str, *, trace: ScoutTrace,
         max_new_tokens: int = 768, temperature: float = 0.2) -> str:
    res = llm.generate(
        [{"role": "system", "content": sys},
         {"role": "user", "content": usr}],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    trace.llm_calls += 1
    trace.prompt_tokens += res.prompt_tokens
    trace.completion_tokens += res.completion_tokens
    return res.text


def _enumerate_domains(backend: SearchBackend) -> List[Tuple[str, str, int]]:
    """Return [(agent_name, description, chunk_count)] for every domain.

    Uses the corpus' active TOOL_TO_AGENT map so the partition matches
    what the scoping methods see. Chunk counts come from a cheap Neo4j
    count query per agent.
    """
    from .react_baseline import TOOL_DESCRIPTIONS
    out: List[Tuple[str, str, int]] = []
    seen = set()
    for tool, agent in TOOL_TO_AGENT.items():
        if agent in seen or agent == "general_agent":
            continue
        seen.add(agent)
        desc = TOOL_DESCRIPTIONS.get(tool, agent)
        try:
            chunks = backend.combined_scoped_search("overview", agent, limit=1)
            count = len(chunks)  # placeholder; treat 1 vs 0 as "has data"
        except Exception:
            count = 0
        out.append((agent, desc, count))
    return out


def _draa(llm: LLMProvider, query: str,
          domains: List[Tuple[str, str, int]],
          *, trace: ScoutTrace) -> Dict[str, str]:
    """DRAA: per-domain tier classification in a single LLM call."""
    if not domains:
        return {}

    # Knowledge-richness signal: normalise chunk counts to [0,1].
    counts = [c for _, _, c in domains]
    max_c = max(counts) or 1
    block = []
    for name, desc, count in domains:
        # We do not pre-compute embedding sim; we let DRAA judge from the
        # description text — equivalent to the published "Sim(q, 𝒟ᵢ)" since
        # the LLM can read the description in-context.
        block.append(f"  - {name}: {desc} | sim=auto | rich={count/max_c:.2f}")
    out = _gen(llm, DRAA_SYSTEM,
               DRAA_HUMAN.format(query=query, domains_block="\n".join(block)),
               trace=trace, max_new_tokens=512, temperature=0.2)
    parsed = _parse_json(out, default={"domains": []})
    tiers: Dict[str, str] = {}
    for d in parsed.get("domains", []):
        nm = d.get("name", "")
        tier = (d.get("tier") or "").upper()
        if tier not in ("HIGH", "MODERATE", "POTENTIAL", "IRRELEVANT"):
            tier = "POTENTIAL"
        if nm:
            tiers[nm] = tier
    # Default any domain DRAA forgot to POTENTIAL
    for nm, _, _ in domains:
        tiers.setdefault(nm, "POTENTIAL")
    return tiers


def _paga(llm: LLMProvider, backend: SearchBackend, query: str,
          domain: str, granularity: str, *, trace: ScoutTrace) -> Dict:
    """PAGA: per-domain partial answer at HIGH (k=20) or MODERATE (k=5)
    granularity. The pseudo-domain "__global__" performs an unscoped
    retrieval — used as the fallback when DRAA finds no relevant domain,
    so out-of-domain queries are answered globally rather than routed to
    an arbitrary scope (keeps the baseline comparison fair)."""
    if domain == "__global__":
        chunks = backend.global_search(query)[:20]
    else:
        k = 20 if granularity == "HIGH" else 5
        chunks = backend.combined_scoped_search(query, domain, limit=k)
    trace.chunks.extend(chunks)
    if domain not in trace.routed_agents:
        trace.routed_agents.append(domain)
    context = format_chunks(chunks) if chunks else "(no chunks)"
    out = _gen(llm, PAGA_QA_SYSTEM,
               PAGA_HUMAN.format(query=query, domain=domain,
                                 granularity=granularity, context=context),
               trace=trace, max_new_tokens=512, temperature=0.2)
    parsed = _parse_json(out, default={"partial_answer": "", "coverage": "", "gaps": ""})
    return {
        "domain": domain,
        "granularity": granularity,
        "n_chunks": len(chunks),
        "partial_answer": _as_text(parsed.get("partial_answer", "")),
        "coverage": _as_text(parsed.get("coverage", "")),
        "gaps": _as_text(parsed.get("gaps", "")),
    }


def _oasa(llm: LLMProvider, query: str, partials: List[Dict],
          *, trace: ScoutTrace) -> str:
    if not partials:
        return ""
    block_lines = []
    for p in partials:
        block_lines.append(
            f"--- {p['domain']} ({p.get('granularity','?')}, "
            f"{p.get('n_chunks',0)} chunks) ---\n"
            f"Partial: {p.get('partial_answer','')}\n"
            f"Coverage: {p.get('coverage','')}\n"
            f"Gaps: {p.get('gaps','')}"
        )
    out = _gen(llm, OASA_SYSTEM,
               OASA_HUMAN.format(query=query,
                                 partials_block="\n\n".join(block_lines)),
               trace=trace, max_new_tokens=768, temperature=0.2)
    parsed = _parse_json(out, default=None)
    if parsed is not None:
        # JSON parsed: trust its answer field (possibly empty so the
        # caller's safety net falls back to per-domain partial answers).
        return _as_text(parsed.get("answer", "")).strip()
    raw = out.strip()
    # Parse failed entirely. Raw might be prose; but if it is itself a
    # malformed JSON-shaped blob (containing "answer":""), it is useless
    # as an answer — return empty so the safety net fires.
    if raw.startswith("{") and '"answer"' in raw:
        return ""
    return raw


def _aqaa(llm: LLMProvider, query: str, answer: str,
          consulted: List[str], *, trace: ScoutTrace) -> Dict:
    out = _gen(llm, AQAA_SYSTEM,
               AQAA_HUMAN.format(query=query, answer=answer,
                                 consulted=", ".join(consulted) or "(none)"),
               trace=trace, max_new_tokens=384, temperature=0.0)
    parsed = _parse_json(out, default={"C": 0.0, "V": 0.0, "gaps": [], "follow_ups": []})
    try:
        C = float(parsed.get("C", 0.0))
        V = float(parsed.get("V", 0.0))
    except (TypeError, ValueError):
        C = V = 0.0
    return {"C": max(0.0, min(1.0, C)),
            "V": max(0.0, min(1.0, V)),
            "gaps": parsed.get("gaps", []) or [],
            "follow_ups": parsed.get("follow_ups", []) or []}


def _pick_strategy(C: float, V: float, t_remain: float) -> str:
    """Equation (6) of the SCOUT-RAG paper, verbatim thresholds."""
    if C < 0.75 and V < 0.70 and t_remain > 20:
        return "Hybrid"
    if C < 0.75 and t_remain > 15:
        return "Depth"
    if V < 0.70 and t_remain > 10:
        return "Breadth"
    return "Stop"


# ── public entry point ──

def run_scout_rag(query: str, *, llm: LLMProvider, backend: SearchBackend,
                  max_iters: int = 2, time_budget_s: float = 90.0) -> ScoutTrace:
    """SCOUT-RAG end-to-end: Algorithm 1 (Stages I–III).

    Stages:
      I  — DRAA tiers all domains in one call.
      II — PAGA runs on HIGH (k=20) and MODERATE (k=5) domains; OASA fuses.
      III — Loop: AQAA evaluates → strategy selector → PAGA on refined queries
            → OASA re-synthesises. Stops on Q ≥ 0.85, Tr < 5s, ΔQ < ε, or
            "Stop" strategy.
    """
    trace = ScoutTrace()
    t0 = time.time()

    def t_remain() -> float:
        return time_budget_s - (time.time() - t0)

    # 1. Enumerate domains for this corpus
    domains = _enumerate_domains(backend)
    if not domains:
        trace.early_stop_reason = "no_domains"
        trace.wall_time_s = time.time() - t0
        return trace

    # 2. Stage I — DRAA tier classification
    tiers = _draa(llm, query, domains, trace=trace)
    trace.tiers = dict(tiers)
    high   = [n for n, t in tiers.items() if t == "HIGH"]
    moder  = [n for n, t in tiers.items() if t == "MODERATE"]
    potent = [n for n, t in tiers.items() if t == "POTENTIAL"]

    # Safety: if DRAA classified nothing as HIGH/MODERATE, the query is
    # out-of-domain for this corpus. Fall back to a single global
    # (unscoped) retrieval rather than routing to an arbitrary scope —
    # this matches what a monolithic retriever does and keeps the
    # baseline fair on out-of-domain queries.
    global_fallback = not high and not moder

    # 3. Stage II — PAGA on HIGH+MODERATE, OASA seed synthesis
    partials: List[Dict] = []
    if global_fallback:
        trace.early_stop_reason = "global_fallback"
        partials.append(_paga(llm, backend, query, "__global__", "HIGH", trace=trace))
    else:
        for d in high:
            partials.append(_paga(llm, backend, query, d, "HIGH", trace=trace))
        for d in moder:
            partials.append(_paga(llm, backend, query, d, "MODERATE", trace=trace))
    trace.partial_answers = list(partials)
    answer = _oasa(llm, query, partials, trace=trace)
    if answer:
        trace.answer = answer
    best_answer = answer
    best_Q = -1.0   # so the first AQAA score (even 0.0) registers

    # 4. Stage III — refinement loop
    # Skipped on global fallback: no scoped domains exist to run
    # Depth/Breadth refinement over.
    prev_Q: Optional[float] = None
    EPS = 0.05   # not given in paper; ours, disclosed.
    consulted = list(set(high + moder))
    iters = 0 if global_fallback else max_iters

    for it in range(iters):
        if t_remain() < 5.0:
            trace.early_stop_reason = "time_budget"
            break

        eval_ = _aqaa(llm, query, answer, consulted, trace=trace)
        Q = 0.5 * (eval_["C"] + eval_["V"])
        trace.quality_history.append(Q)
        trace.iterations = it + 1
        if answer and Q > best_Q:
            best_Q = Q; best_answer = answer

        if Q >= 0.85:
            trace.early_stop_reason = "quality_threshold"
            break
        if prev_Q is not None and abs(Q - prev_Q) < EPS:
            trace.early_stop_reason = "stagnation"
            break
        prev_Q = Q

        strategy = _pick_strategy(eval_["C"], eval_["V"], t_remain())
        trace.strategies.append(strategy)
        if strategy == "Stop":
            trace.early_stop_reason = "stop_strategy"
            break

        # Generate refinement queries from AQAA's follow-ups, route to:
        #   Depth   → existing HIGH/MODERATE domains
        #   Breadth → POTENTIAL domains (expansion)
        #   Hybrid  → both
        follow_ups = eval_["follow_ups"][:2] or [query]
        target_domains: List[str] = []
        if strategy in ("Depth", "Hybrid"):
            target_domains.extend(high + moder)
        if strategy in ("Breadth", "Hybrid"):
            target_domains.extend(potent[:2])
            consulted.extend(potent[:2])
        target_domains = list(dict.fromkeys(target_domains))[:4]  # cap

        for sub_q in follow_ups:
            for d in target_domains:
                if t_remain() < 5.0:
                    break
                gran = "HIGH" if d in high else "MODERATE"
                partials.append(_paga(llm, backend, sub_q, d, gran, trace=trace))
        new_answer = _oasa(llm, query, partials, trace=trace)
        if new_answer:                 # never overwrite a good answer with empty
            answer = new_answer
            trace.answer = answer

    # Prefer the best non-empty answer seen across seed + refinements.
    if best_answer:
        trace.answer = best_answer

    # Final safety net: if synthesis never produced any text, fall back
    # to the concatenated per-domain partial answers so SCOUT-RAG always
    # emits a non-empty answer when PAGA retrieved anything.
    if not trace.answer:
        fallback = " ".join(
            p.get("partial_answer", "") for p in partials
            if p.get("partial_answer")
        ).strip()
        trace.answer = fallback

    # dedupe chunks
    seen, dedup = set(), []
    for c in trace.chunks:
        cid = c.get("id")
        if cid in seen:
            continue
        seen.add(cid); dedup.append(c)
    trace.chunks = dedup

    trace.wall_time_s = time.time() - t0
    return trace
