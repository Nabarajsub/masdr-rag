"""
Composite enterprise corpus eval runner.

Uses the LocalSearchBackend (FAISS in-memory) instead of Neo4j, so this
benchmark stays entirely on the cluster filesystem.

Usage:
    python -m emnlp_evaluation.runners.run_composite \\
        --llm qwen --limit 30 --systems hybrid_routed,masdr_rag,react
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO.parent))

from emnlp_evaluation.configs import oss_config as cfg
from emnlp_evaluation.llm_providers import get_provider, ToolSpec
from emnlp_evaluation.embeddings import get_embedder
from emnlp_evaluation.composite_corpus.local_backend import load_backend
from emnlp_evaluation.agents.orchestrator_oss import run_orchestrator
from emnlp_evaluation.agents.orchestrator_singlecall import run_singlecall_orchestrator
from emnlp_evaluation.agents.hybrid_routed import run_hybrid_routed, run_regex_scoped
from emnlp_evaluation.agents.naive_rag import run_naive_rag
from emnlp_evaluation.agents.react_baseline import run_react
from emnlp_evaluation.agents.ma_rag import run_ma_rag
from emnlp_evaluation.agents.scout_rag import run_scout_rag


COMPOSITE_PREFIX = str(_REPO / "emnlp_evaluation" / "composite_corpus" / "data" / "composite")
QUERIES_PATH = _REPO / "emnlp_evaluation" / "composite_corpus" / "data" / "queries.json"


# ── source-type → scoped agent map (one agent per source type) ─────────────

_SOURCE_TYPES = [
    "gmail", "slack", "github", "jira", "confluence",
    "docs", "stackoverflow", "helpdesk", "reports",
]

_TOOL_DESCRIPTIONS = {
    "search_gmail":         "Search Enron-style email correspondence (gmail).",
    "search_slack":         "Search Slack-style multi-turn chat threads.",
    "search_github":        "Search GitHub issue and PR discussions.",
    "search_jira":          "Search Jira-style bug-tracker tickets.",
    "search_confluence":    "Search wiki / Confluence-style internal pages.",
    "search_docs":          "Search product / technical documentation passages.",
    "search_stackoverflow": "Search Stack Overflow-style developer Q&A.",
    "search_helpdesk":      "Search customer-service / helpdesk conversations.",
    "search_reports":       "Search financial reports and SEC filings.",
    "search_general":       "Search across ALL enterprise sources (fallback).",
}


# Regex fast-path for queries that explicitly name their source. Most
# Composite-9 queries do not, so the regex tier falls through: regex_scoped
# degrades to global search (reported as-is) and hybrid_routed falls back to
# the LLM router, which routes on the per-source descriptions.
_COMPOSITE_REGEX_RULES = [
    (r"\b(email|inbox|correspondence)\b",                     "gmail_agent"),
    (r"\b(slack|chat thread|channel)\b",                      "slack_agent"),
    (r"\b(github|pull request|\bPR\b)\b",                     "github_agent"),
    (r"\b(jira|bug ticket|sprint)\b",                         "jira_agent"),
    (r"\b(confluence|wiki)\b",                                "confluence_agent"),
    (r"\b(stack ?overflow)\b",                                "stackoverflow_agent"),
    (r"\b(helpdesk|help desk|support ticket)\b",              "helpdesk_agent"),
    (r"\b(sec filing|10-k|annual report|financial report)\b", "reports_agent"),
    (r"\b(documentation|user guide|technical doc)\b",         "docs_agent"),
]


def _build_catalog():
    """Install a per-source tool catalog into the shared modules."""
    from emnlp_evaluation.agents import tool_catalog as tc
    from emnlp_evaluation.agents import react_baseline as rb
    from emnlp_evaluation.agents import hybrid_routed as hr

    params = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Search query."}},
        "required": ["query"],
    }
    tools: List[ToolSpec] = []
    tool_to_agent: Dict[str, str] = {}
    source_filters: Dict[str, List[str]] = {}
    for st in _SOURCE_TYPES:
        tname = f"search_{st}"
        aname = f"{st}_agent"
        tools.append(ToolSpec(name=tname, description=_TOOL_DESCRIPTIONS[tname], parameters=params))
        tool_to_agent[tname] = aname
        source_filters[aname] = [st]
    tools.append(ToolSpec(
        name="search_general",
        description=_TOOL_DESCRIPTIONS["search_general"],
        parameters=params,
    ))
    tool_to_agent["search_general"] = "general_agent"

    tc.TOOL_TO_AGENT.clear(); tc.TOOL_TO_AGENT.update(tool_to_agent)
    # The orchestrator imports build_tool_catalog at module load, so monkeypatching
    # tc.build_tool_catalog alone has no effect. The original function iterates
    # tc.TOOL_TO_AGENT keys against tc._TOOL_DESCRIPTIONS — we must also replace
    # that dict so the original function returns the composite tools.
    tc._TOOL_DESCRIPTIONS.clear()
    for t in tools:
        tc._TOOL_DESCRIPTIONS[t.name] = t.description
    tc.build_tool_catalog = lambda: tools  # type: ignore[assignment]
    rb.TOOL_DESCRIPTIONS.clear()
    for t in tools:
        rb.TOOL_DESCRIPTIONS[t.name] = t.description
    hr.configure_router(
        regex_rules=_COMPOSITE_REGEX_RULES,
        domain_desc=("an enterprise knowledge base with nine distinct data "
                     "sources (email, chat, code review, tickets, wiki, docs, "
                     "developer Q&A, helpdesk, financial reports)"),
    )
    return source_filters


def _trace_dict(trace, system: str, **extras):
    d = {
        "system": system,
        "answer": trace.answer,
        "n_chunks": len(trace.chunks),
        "chunk_ids": [c.get("id") for c in trace.chunks],
        "chunk_sources": [c.get("source") for c in trace.chunks],
        "chunk_titles": [c.get("title") for c in trace.chunks],
        "prompt_tokens": trace.prompt_tokens,
        "completion_tokens": trace.completion_tokens,
        "llm_calls": trace.llm_calls,
        "wall_time_s": trace.wall_time_s,
    }
    d.update(extras)
    return d


def _run_one(system: str, query: str, llm, backend):
    if system in ("monolithic", "naive"):
        t = run_naive_rag(query, llm=llm, backend=backend); return _trace_dict(t, system)
    if system == "regex_scoped":
        t = run_regex_scoped(query, llm=llm, backend=backend)
        return _trace_dict(t, system, routed_agent=t.routed_agent, route_decision=t.route_decision)
    if system == "hybrid_routed":
        t = run_hybrid_routed(query, llm=llm, backend=backend)
        return _trace_dict(t, system, routed_agent=t.routed_agent, route_decision=t.route_decision)
    if system == "masdr_rag":
        t = run_orchestrator(query, llm=llm, backend=backend)
        return _trace_dict(t, system, routed_agents=t.routed_agents, tool_calls=t.tool_calls)
    if system == "masdr_singlecall":
        t = run_singlecall_orchestrator(query, llm=llm, backend=backend)
        return _trace_dict(t, system, routed_agents=t.routed_agents, tool_calls=t.tool_calls)
    if system == "react":
        t = run_react(query, llm=llm, backend=backend)
        return _trace_dict(t, system, routed_agents=t.routed_agents,
                           tool_calls=t.actions, iterations=t.iterations)
    if system == "ma_rag":
        t = run_ma_rag(query, llm=llm, backend=backend)
        return _trace_dict(t, system, plan=t.plan,
                           step_outputs=t.step_outputs)
    if system == "scout_rag":
        t = run_scout_rag(query, llm=llm, backend=backend)
        return _trace_dict(t, system, routed_agents=t.routed_agents,
                           tiers=t.tiers, strategies=t.strategies,
                           quality_history=t.quality_history,
                           iterations=t.iterations,
                           early_stop_reason=t.early_stop_reason)
    raise ValueError(system)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="monolithic,hybrid_routed,masdr_rag,react")
    ap.add_argument("--llm", default="qwen", choices=cfg.SUPPORTED_PROVIDERS)
    ap.add_argument("--prefix", default=COMPOSITE_PREFIX)
    ap.add_argument("--queries", default=str(QUERIES_PATH))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    systems = [s.strip() for s in args.systems.split(",")]
    source_filters = _build_catalog()

    llm = get_provider(args.llm)
    embedder = get_embedder("bge_m3")
    backend = load_backend(args.prefix, embedder=embedder,
                            scope_field="source_type", source_filters=source_filters)

    with open(args.queries) as f:
        suite = json.load(f)
    if args.limit:
        suite = suite[: args.limit]

    out_path = Path(args.out) if args.out else (
        cfg.RESULTS_DIR / f"composite_{args.llm}_{'-'.join(systems)}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "a") as f:
        for q in suite:
            qid = q["query_id"]
            for system in systems:
                t0 = time.time()
                try:
                    rec = _run_one(system, q["query"], llm, backend)
                    rec.update({
                        "query_id": qid, "query": q["query"],
                        "reference_answer": q.get("reference_answer", ""),
                        "category": q.get("gold_source_type"),
                        "query_type": q.get("query_type"),
                        "gold_doc_id": q.get("gold_doc_id"),
                        "gold_chunk_id": q.get("gold_chunk_id"),
                        "llm": args.llm, "embedder": "bge_m3",
                        "benchmark": "composite",
                    })
                except Exception as e:
                    rec = {"query_id": qid, "system": system,
                           "error": f"{type(e).__name__}: {e}",
                           "llm": args.llm, "embedder": "bge_m3",
                           "benchmark": "composite"}
                f.write(json.dumps(rec) + "\n"); f.flush()
                print(f"[{qid:>10}] [{system:<14}] {time.time()-t0:5.1f}s "
                      f"chunks={rec.get('n_chunks','-')}", flush=True)

    print(f"[composite] wrote {out_path}")


if __name__ == "__main__":
    main()
