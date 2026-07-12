"""MultiHop-RAG eval runner.

Same architecture as run_composite.py: load LocalSearchBackend from the
multihop.embeddings.npy + meta.parquet built in
emnlp_evaluation.benchmark.multihop_rag.build_faiss, install a per-category
tool catalog (business/technology/sports/entertainment/science/health +
general), and run any of the five systems.

Usage:
    python -m emnlp_evaluation.runners.run_multihop \\
        --llm qwen --systems monolithic,regex_scoped,hybrid_routed,masdr_rag \\
        --limit 200
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
from emnlp_evaluation.composite_corpus.local_backend import load_backend
from emnlp_evaluation.agents.orchestrator_oss import run_orchestrator
from emnlp_evaluation.agents.orchestrator_singlecall import run_singlecall_orchestrator
from emnlp_evaluation.agents.hybrid_routed import run_hybrid_routed, run_regex_scoped
from emnlp_evaluation.agents.naive_rag import run_naive_rag
from emnlp_evaluation.agents.react_baseline import run_react


PREFIX = str(_REPO / "emnlp_evaluation" / "benchmark" / "multihop_rag" / "data" / "multihop")
QUERIES = _REPO / "emnlp_evaluation" / "benchmark" / "multihop_rag" / "data" / "queries.json"

CATEGORIES = ["business", "entertainment", "sports", "technology", "science", "health"]
_DESC = {
    "search_business":      "Search business and finance news articles.",
    "search_entertainment": "Search entertainment, celebrity, and pop-culture news.",
    "search_sports":        "Search sports news and game coverage.",
    "search_technology":    "Search technology and product news.",
    "search_science":       "Search science and research news.",
    "search_health":        "Search health and medicine news.",
    "search_general":       "Search ALL news articles regardless of category (fallback).",
}


def _build_catalog():
    from emnlp_evaluation.agents import tool_catalog as tc
    from emnlp_evaluation.agents import react_baseline as rb

    params = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Search query."}},
        "required": ["query"],
    }
    tools: List[ToolSpec] = []
    tool_to_agent: Dict[str, str] = {}
    source_filters: Dict[str, List[str]] = {}
    for cat in CATEGORIES:
        tname, aname = f"search_{cat}", f"{cat}_agent"
        tools.append(ToolSpec(name=tname, description=_DESC[tname], parameters=params))
        tool_to_agent[tname] = aname
        source_filters[aname] = [cat]
    tools.append(ToolSpec(name="search_general", description=_DESC["search_general"], parameters=params))
    tool_to_agent["search_general"] = "general_agent"

    tc.TOOL_TO_AGENT.clear(); tc.TOOL_TO_AGENT.update(tool_to_agent)
    tc._TOOL_DESCRIPTIONS.clear()
    for t in tools:
        tc._TOOL_DESCRIPTIONS[t.name] = t.description
    tc.build_tool_catalog = lambda: tools
    rb.TOOL_DESCRIPTIONS.clear()
    for t in tools:
        rb.TOOL_DESCRIPTIONS[t.name] = t.description
    return source_filters


def _trace_dict(trace, system: str, **extras):
    d = {
        "system": system, "answer": trace.answer,
        "n_chunks": len(trace.chunks),
        "chunk_ids": [c.get("id") for c in trace.chunks],
        "chunk_titles": [c.get("title") for c in trace.chunks],
        "chunk_series": [c.get("title") for c in trace.chunks],
        "chunk_sources": [c.get("source", "") for c in trace.chunks],
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
    raise ValueError(system)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="monolithic,regex_scoped,hybrid_routed,masdr_rag")
    ap.add_argument("--llm", default="qwen",
                    help="qwen/llama/gemini or openrouter:<slug>")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    source_filters = _build_catalog()

    if args.llm.startswith("openrouter:"):
        llm = get_provider("openrouter", model_name=args.llm.split(":", 1)[1])
    else:
        llm = get_provider(args.llm)
    embedder = get_embedder("bge_m3")
    backend = load_backend(PREFIX, embedder=embedder,
                            scope_field="source_type", source_filters=source_filters)

    suite = json.load(open(QUERIES))
    if args.limit:
        suite = suite[: args.limit]

    llm_tag = args.llm.replace("openrouter:", "or_").replace("/", "-")
    out_path = Path(args.out) if args.out else (
        cfg.RESULTS_DIR / f"multihop_{llm_tag}_{'-'.join(systems)}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support
    done = set()
    if out_path.exists():
        for line in open(out_path):
            try:
                r = json.loads(line)
                if "error" not in r:
                    done.add((r.get("query_id"), r.get("system")))
            except json.JSONDecodeError:
                continue
        if done:
            print(f"[runner] resume: {len(done)} already present")

    with open(out_path, "a") as f:
        for q in suite:
            qid = q["query_id"]
            for system in systems:
                if (qid, system) in done:
                    continue
                t0 = time.time()
                try:
                    rec = _run_one(system, q["query"], llm, backend)
                    rec.update({
                        "query_id": qid, "query": q["query"],
                        "reference_answer": q.get("reference_answer", ""),
                        "category": q.get("category"),
                        "question_type": q.get("question_type"),
                        "gold_titles": q.get("gold_titles", []),
                        "llm": args.llm, "embedder": "bge_m3",
                        "benchmark": "multihop_rag",
                    })
                except Exception as e:
                    rec = {"query_id": qid, "system": system,
                           "error": f"{type(e).__name__}: {e}",
                           "llm": args.llm, "embedder": "bge_m3",
                           "benchmark": "multihop_rag"}
                f.write(json.dumps(rec) + "\n"); f.flush()
                print(f"[{qid:>12}] [{system:<14}] {time.time()-t0:5.1f}s "
                      f"chunks={rec.get('n_chunks','-')}", flush=True)

    print(f"[multihop] wrote {out_path}")


if __name__ == "__main__":
    main()
