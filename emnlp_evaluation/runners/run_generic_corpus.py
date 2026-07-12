"""Generic eval runner for any corpus with FAISS-style local artifacts.

A small step beyond run_composite/run_multihop: a single runner that takes
a benchmark name (financebench/mmlu_pro/nq) and dispatches to the right
prefix + per-corpus scope vocabulary.

Usage:
    python -m emnlp_evaluation.runners.run_generic_corpus \\
        --benchmark mmlu_pro --llm qwen \\
        --systems monolithic,regex_scoped,hybrid_routed,masdr_rag --limit 500
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
from emnlp_evaluation.composite_corpus.local_neo4j_backend import load_neo4j_backend
from emnlp_evaluation.composite_corpus.rerank_backend import RerankBackend
from emnlp_evaluation.composite_corpus.splade_backend import load_splade_backend
from emnlp_evaluation.agents.orchestrator_oss import run_orchestrator
from emnlp_evaluation.agents.hybrid_routed import run_hybrid_routed, run_regex_scoped
from emnlp_evaluation.agents.naive_rag import run_naive_rag
from emnlp_evaluation.agents.ma_rag import run_ma_rag
from emnlp_evaluation.agents.scout_rag import run_scout_rag


BENCH = {
    "financebench": {
        "prefix": _REPO / "emnlp_evaluation/benchmark/financebench/data/financebench",
        "queries": _REPO / "emnlp_evaluation/benchmark/financebench/data/queries.json",
        "scope_field": "source_type",
    },
    "mmlu_pro": {
        "prefix": _REPO / "emnlp_evaluation/benchmark/mmlu_pro/data/mmlu_pro",
        "queries": _REPO / "emnlp_evaluation/benchmark/mmlu_pro/data/queries.json",
        "scope_field": "source_type",
    },
    "nq": {
        "prefix": _REPO / "emnlp_evaluation/benchmark/nq/data/nq",
        "queries": _REPO / "emnlp_evaluation/benchmark/nq/data/queries.json",
        "scope_field": "source_type",
    },
    "multihop": {
        "prefix": _REPO / "emnlp_evaluation/benchmark/multihop_rag/data/multihop",
        "queries": _REPO / "emnlp_evaluation/benchmark/multihop_rag/data/queries.json",
        "scope_field": "source_type",
    },
    "composite": {
        "prefix": _REPO / "emnlp_evaluation/composite_corpus/data/composite",
        "queries": _REPO / "emnlp_evaluation/composite_corpus/data/queries.json",
        "scope_field": "source_type",
    },
}


def _build_catalog(scopes: List[str], benchmark: str = "the corpus"):
    """Build a per-scope routing catalog for whatever vocabulary the corpus uses."""
    import re as _re
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
    for s in scopes:
        tname = f"search_{s}"
        aname = f"{s}_agent"
        tools.append(ToolSpec(
            name=tname,
            description=f"Search content in the '{s}' subset of the corpus.",
            parameters=params,
        ))
        tool_to_agent[tname] = aname
        source_filters[aname] = [s]
    tools.append(ToolSpec(
        name="search_general",
        description="Search ALL chunks regardless of scope (fallback).",
        parameters=params,
    ))
    tool_to_agent["search_general"] = "general_agent"

    tc.TOOL_TO_AGENT.clear(); tc.TOOL_TO_AGENT.update(tool_to_agent)
    tc._TOOL_DESCRIPTIONS.clear()
    for t in tools:
        tc._TOOL_DESCRIPTIONS[t.name] = t.description
    tc.build_tool_catalog = lambda: tools
    rb.TOOL_DESCRIPTIONS.clear()
    for t in tools:
        rb.TOOL_DESCRIPTIONS[t.name] = t.description
    # Regex fast-path: match a scope only when the query names it literally
    # (e.g. a publisher or subject); everything else falls through to the LLM
    # router, which sees the per-scope descriptions.
    hr.configure_router(
        regex_rules=[(rf"\b{_re.escape(s).replace(chr(92)+'_', '[ _-]')}\b", f"{s}_agent")
                     for s in scopes],
        domain_desc=(f"a '{benchmark}' corpus partitioned into "
                     f"{len(scopes)} scopes: {', '.join(scopes)}"),
    )
    return source_filters


def _trace_dict(trace, system: str, **extras):
    d = {
        "system": system, "answer": trace.answer,
        "n_chunks": len(trace.chunks),
        "chunk_ids": [c.get("id") for c in trace.chunks],
        "chunk_titles": [c.get("title") for c in trace.chunks],
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
    if system == "ma_rag":
        t = run_ma_rag(query, llm=llm, backend=backend)
        return _trace_dict(t, system, plan=t.plan, step_outputs=t.step_outputs)
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
    ap.add_argument("--benchmark", required=True, choices=list(BENCH))
    ap.add_argument("--systems", default="monolithic,regex_scoped,hybrid_routed,masdr_rag")
    ap.add_argument("--llm", default="qwen",
                    help="qwen/llama/gemini or openrouter:<slug>")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--backend", choices=("faiss", "neo4j", "splade"), default="faiss",
                    help="faiss = local FAISS IndexFlatIP (default); "
                         "neo4j = local Neo4j HNSW index "
                         "(corpus must be pre-ingested via ingest_to_local_neo4j.py); "
                         "splade = sparse SPLADE matrix in <prefix>.splade.npz.")
    ap.add_argument("--rerank", action="store_true",
                    help="Wrap the backend with BGE-reranker-v2-m3 cross-encoder. "
                         "Pulls top-30 bi-encoder candidates, rescores, returns top-10.")
    args = ap.parse_args()

    spec = BENCH[args.benchmark]
    suite = json.load(open(spec["queries"]))
    if args.limit:
        suite = suite[: args.limit]
    print(f"[{args.benchmark}] {len(suite)} queries")

    # Discover scope vocabulary from the meta parquet
    import pandas as pd
    meta = pd.read_parquet(f"{spec['prefix']}.meta.parquet")
    scopes = sorted(meta[spec["scope_field"]].dropna().astype(str).unique())
    print(f"[{args.benchmark}] {len(scopes)} scopes: {scopes}")
    source_filters = _build_catalog(scopes, benchmark=args.benchmark)

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    if args.llm.startswith("openrouter:"):
        llm = get_provider("openrouter", model_name=args.llm.split(":", 1)[1])
    else:
        llm = get_provider(args.llm)
    embedder = get_embedder("bge_m3")
    if args.backend == "neo4j":
        backend = load_neo4j_backend(args.benchmark, embedder=embedder,
                                      scope_field=spec["scope_field"],
                                      source_filters=source_filters)
    elif args.backend == "splade":
        backend = load_splade_backend(str(spec["prefix"]),
                                       source_filters=source_filters,
                                       scope_field=spec["scope_field"])
    else:
        backend = load_backend(spec["prefix"], embedder=embedder,
                                scope_field=spec["scope_field"],
                                source_filters=source_filters)
    if args.rerank:
        backend = RerankBackend(backend)

    llm_tag = args.llm.replace("openrouter:", "or_").replace("/", "-")
    backend_tag = f"_{args.backend}" if args.backend != "faiss" else ""
    rerank_tag = "_rerank" if args.rerank else ""
    out_path = Path(args.out) if args.out else (
        cfg.RESULTS_DIR / f"{args.benchmark}_{llm_tag}{backend_tag}{rerank_tag}_{'-'.join(systems)}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out_path.exists():
        for line in open(out_path):
            try:
                r = json.loads(line)
                if "error" not in r:
                    done.add((r.get("query_id"), r.get("system")))
            except json.JSONDecodeError:
                continue

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
                        "gold_chunk_id": q.get("gold_chunk_id", ""),
                        "gold_chunk_ids": q.get("gold_chunk_ids", []),
                        "gold_titles": q.get("gold_titles", []),
                        "gold_short_answers": q.get("gold_short_answers", []),
                        "llm": args.llm, "embedder": "bge_m3",
                        "benchmark": args.benchmark,
                    })
                except Exception as e:
                    rec = {"query_id": qid, "system": system,
                           "error": f"{type(e).__name__}: {e}",
                           "llm": args.llm, "embedder": "bge_m3",
                           "benchmark": args.benchmark}
                f.write(json.dumps(rec) + "\n"); f.flush()
                print(f"[{qid:>14}] [{system:<14}] {time.time()-t0:5.1f}s "
                      f"chunks={rec.get('n_chunks','-')}", flush=True)

    print(f"[{args.benchmark}] wrote {out_path}")


if __name__ == "__main__":
    main()
