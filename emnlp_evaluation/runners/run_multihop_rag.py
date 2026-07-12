"""
MultiHop-RAG evaluation runner.

Mirrors run_wydot_oss.py but uses the MultiHop subgraph and a per-publisher
tool catalog (one tool per discovered source).
"""
from __future__ import annotations

import argparse
import json
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
from emnlp_evaluation.benchmark.multihop_rag.multihop_backend import make_multihop_backend
from emnlp_evaluation.benchmark.multihop_rag.source_map import discover_sources, build_source_filters, _slug
from emnlp_evaluation.agents.orchestrator_oss import run_orchestrator
from emnlp_evaluation.agents.hybrid_routed import run_hybrid_routed, run_regex_scoped
from emnlp_evaluation.agents.naive_rag import run_naive_rag
from emnlp_evaluation.agents.react_baseline import run_react
from emnlp_evaluation.agents.ma_rag import run_ma_rag
from emnlp_evaluation.agents.scout_rag import run_scout_rag


SUITE_PATH = Path(__file__).resolve().parents[1] / "benchmark" / "multihop_rag" / "data" / "MultiHopRAG.json"


def _build_multihop_tool_catalog(sources: List[str]):
    """Per-source tools shaped like WYDOT's scoped tools so the same orchestrator works."""
    from emnlp_evaluation.agents import tool_catalog as tc
    params = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
        },
        "required": ["query"],
    }
    tools = []
    tool_to_agent = {}
    for src in sources:
        name = f"search_{_slug(src)}"
        agent_name = f"{_slug(src)}_agent"
        tools.append(ToolSpec(
            name=name,
            description=f"Search MultiHop-RAG articles published by '{src}'.",
            parameters=params,
        ))
        tool_to_agent[name] = agent_name
    # Plus a fallback global tool
    tools.append(ToolSpec(
        name="search_general",
        description="Search across ALL MultiHop-RAG sources (use when topic spans multiple publishers).",
        parameters=params,
    ))
    tool_to_agent["search_general"] = "general_agent"
    # Monkey-patch the tool_catalog module so orchestrator/react use the new map.
    tc.TOOL_TO_AGENT.clear(); tc.TOOL_TO_AGENT.update(tool_to_agent)
    # Also patch the build_tool_catalog used by run_orchestrator.
    tc.build_tool_catalog = lambda: tools  # type: ignore[assignment]
    # And ReAct's TOOL_DESCRIPTIONS so it lists the right tools.
    from emnlp_evaluation.agents import react_baseline as rb
    rb.TOOL_DESCRIPTIONS.clear()
    for t in tools:
        rb.TOOL_DESCRIPTIONS[t.name] = t.description
    return tools, tool_to_agent


def _load_queries(suite_path: Path, limit: int | None):
    with open(suite_path) as f:
        items = json.load(f)
    if limit:
        items = items[:limit]
    return items


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
    if system == "hybrid_routed":
        t = run_hybrid_routed(query, llm=llm, backend=backend)
        return _trace_dict(t, system, routed_agent=t.routed_agent,
                           route_decision=t.route_decision)
    if system == "masdr_rag":
        t = run_orchestrator(query, llm=llm, backend=backend)
        return _trace_dict(t, system, routed_agents=t.routed_agents,
                           tool_calls=t.tool_calls)
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
    raise ValueError(f"Unknown system {system}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="monolithic,hybrid_routed,masdr_rag,react")
    ap.add_argument("--llm", default="qwen", choices=cfg.SUPPORTED_PROVIDERS)
    ap.add_argument("--embedder", default="bge_m3", choices=cfg.SUPPORTED_EMBEDDERS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    systems = [s.strip() for s in args.systems.split(",")]
    sources = discover_sources()
    print(f"[mh-runner] {len(sources)} sources discovered: {sources[:8]}{'...' if len(sources)>8 else ''}")

    llm = get_provider(args.llm)
    embedder = get_embedder(args.embedder)
    backend = make_multihop_backend(embedder)

    _build_multihop_tool_catalog(sources)

    suite = _load_queries(SUITE_PATH, args.limit)
    out_path = Path(args.out) if args.out else (
        cfg.RESULTS_DIR / f"multihop_{args.llm}_{args.embedder}_{'-'.join(systems)}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "a") as f:
        for i, q in enumerate(suite):
            query = q.get("query") or q.get("question", "")
            qid = q.get("id") or q.get("question_id") or f"mh_{i:05d}"
            for system in systems:
                t0 = time.time()
                try:
                    rec = _run_one(system, query, llm, backend)
                    rec.update({
                        "query_id": qid, "query": query,
                        "reference_answer": q.get("answer", ""),
                        "evidence": q.get("evidence_list", q.get("evidence", [])),
                        "query_type": q.get("question_type") or q.get("type"),
                        "llm": args.llm, "embedder": args.embedder,
                        "benchmark": "multihop_rag",
                    })
                except Exception as e:
                    rec = {"query_id": qid, "system": system,
                           "error": f"{type(e).__name__}: {e}",
                           "llm": args.llm, "embedder": args.embedder,
                           "benchmark": "multihop_rag"}
                f.write(json.dumps(rec) + "\n"); f.flush()
                print(f"[{qid:>12}] [{system:<14}] {time.time()-t0:5.1f}s "
                      f"chunks={rec.get('n_chunks','-')}", flush=True)

    print(f"[mh-runner] wrote {out_path}")


if __name__ == "__main__":
    main()
