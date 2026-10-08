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
from emnlp_evaluation.agents.hybrid_routed import (run_hybrid_routed, run_regex_scoped,
                                                   run_r2_routed, run_soft_scoped,
                                                   run_oracle_scoped, run_composed,
                                                   CentroidRouter)
from emnlp_evaluation.agents.naive_rag import run_naive_rag
from emnlp_evaluation.agents.ma_rag import run_ma_rag
from emnlp_evaluation.agents.scout_rag import run_scout_rag


BENCH = {
    "financebench": {
        "prefix": _REPO / "emnlp_evaluation/benchmark/financebench/data/financebench",
        "queries": _REPO / "emnlp_evaluation/benchmark/financebench/data/queries.json",
        "scope_field": "source_type",
    },
    # HotpotQA-distractor, reached through the CRAG harness. run_crag.py owns
    # the tool-catalog arms (it carries hand-written per-bucket tool
    # descriptions and a corpus-specific router config); this entry exists so
    # soft_scoped can run on the same store, since soft scoping routes by
    # embedding centroid and never touches the tool catalog or the LLM router.
    # 54-document subset of WYDOT (16 gold + 38 random distractors, seed 1000),
    # for the interventional dilution experiment: same 32 queries, same gold
    # documents, only the distractor pool differs from the full corpus.
    "wydot54": {
        "prefix": _REPO.parent / "data/wydot/derived/wydot54",
        "queries": _REPO.parent / "data/wydot/derived/wydot54.queries.json",
        "scope_field": "document_series",
    },
    "crag": {
        "prefix": _REPO / "emnlp_evaluation/benchmark/crag/data/crag",
        "queries": _REPO / "emnlp_evaluation/benchmark/crag/data/crag.queries.json",
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
    # Hand-vs-discovered scope experiment: same store (wydotv3 FAISS, BGE-M3,
    # 217,752 chunks), same queries — only the scope taxonomy differs.
    "wydot_hand": {
        "prefix": _REPO.parent / "data/wydot/derived/wydotv3",
        "queries": _REPO.parent / "data/wydot/derived/wydotv3.queries.json",
        "scope_field": "document_series",
        "r2_artifact": _REPO / "emnlp_evaluation/router/artifacts/T2_bge_logreg_hand_series.pkl",
    },
    "wydot_disc": {
        "prefix": _REPO.parent / "data/wydot/derived/wydotv3",
        "queries": _REPO.parent / "data/wydot/derived/wydotv3.queries.json",
        "scope_field": "discovered_scope",
        "r2_artifact": _REPO / "emnlp_evaluation/router/artifacts/T2_bge_logreg_disc_k11.pkl",
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


_R2_ROUTER = None       # set in main() when the benchmark defines an r2_artifact
_CENTROID_ROUTER = None  # set in main() for soft_scoped (faiss backend only)
_SOFT_M = 2
_SOFT_GW = 1.0
_ORACLE_FIELD = None   # query field holding the gold scope (oracle_scoped arm)
_SCOPES: set = set()


def _run_one(system: str, query: str, llm, backend, qrec=None):
    if system == "oracle_scoped":
        gold = str((qrec or {}).get(_ORACLE_FIELD) or "")
        agent = f"{gold}_agent" if gold in _SCOPES and gold != "General" else "general_agent"
        t = run_oracle_scoped(query, agent, llm=llm, backend=backend)
        return _trace_dict(t, system, routed_agent=t.routed_agent, route_decision=t.route_decision)
    if system in ("monolithic", "naive"):
        t = run_naive_rag(query, llm=llm, backend=backend); return _trace_dict(t, system)
    if system.startswith("composed_"):
        t = run_composed(query, llm=llm, backend=backend, centroid_router=_CENTROID_ROUTER,
                         n_scopes=int(system.split("_")[1]))
        return _trace_dict(t, system, routed_agent=t.routed_agent, route_decision=t.route_decision,
                           chunk_scores=[c.get("score") for c in t.chunks])
    if system == "soft_scoped":
        if _CENTROID_ROUTER is None:
            raise RuntimeError("soft_scoped requires the faiss backend (in-memory embeddings)")
        t = run_soft_scoped(query, llm=llm, backend=backend,
                            centroid_router=_CENTROID_ROUTER, m=_SOFT_M,
                            global_weight=_SOFT_GW)
        return _trace_dict(t, system, routed_agent=t.routed_agent,
                           route_decision=t.route_decision)
    if system == "r2_routed":
        if _R2_ROUTER is None:
            raise RuntimeError("r2_routed requires an 'r2_artifact' in the BENCH entry")
        t = run_r2_routed(query, llm=llm, backend=backend, router=_R2_ROUTER)
        return _trace_dict(t, system, routed_agent=t.routed_agent, route_decision=t.route_decision)
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
    ap.add_argument("--soft-m", type=int, default=2,
                    help="soft_scoped: number of centroid-routed scopes to union")
    ap.add_argument("--soft-global-weight", type=float, default=1.0,
                    help="soft_scoped: RRF weight of the global list")
    ap.add_argument("--shared-answer-prompt", action="store_true",
                    help="E5 prompt control: force monolithic AND all scoped systems to use one "
                         "identical, corpus-neutral answer prompt, so the comparison isolates the "
                         "retrieval architecture rather than prompt wording.")
    ap.add_argument("--rerank", action="store_true",
                    help="Wrap the backend with BGE-reranker-v2-m3 cross-encoder. "
                         "Pulls top-30 bi-encoder candidates, rescores, returns top-10.")
    ap.add_argument("--prefix", default=None,
                    help="Override the benchmark's corpus prefix (e.g. an RQ1 subset corpus); "
                         "scope_field is still taken from --benchmark.")
    ap.add_argument("--queries", default=None, help="Override the benchmark's query file.")
    ap.add_argument("--oracle-field", default=None,
                    help="oracle_scoped: query field holding the gold scope (e.g. gold_source_type).")
    args = ap.parse_args()

    spec = dict(BENCH[args.benchmark])
    if args.prefix:
        spec["prefix"] = Path(args.prefix)
    if args.queries:
        spec["queries"] = Path(args.queries)
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
    global _ORACLE_FIELD, _SCOPES
    _ORACLE_FIELD, _SCOPES = args.oracle_field, set(scopes)
    if "oracle_scoped" in args.systems and not args.oracle_field:
        raise SystemExit("oracle_scoped needs --oracle-field")

    # Trained-router arm: labels are scope values; agents are the catalog's
    # f"{scope}_agent" names. 'General' keeps production semantics (global).
    global _R2_ROUTER
    if spec.get("r2_artifact"):
        from emnlp_evaluation.agents.r2_router import R2Router
        label_to_agent = {s: ("general_agent" if s == "General" else f"{s}_agent")
                          for s in scopes}
        _R2_ROUTER = R2Router(spec["r2_artifact"], label_to_agent=label_to_agent)
        print(f"[{args.benchmark}] r2 router: {Path(spec['r2_artifact']).name} "
              f"labels={_R2_ROUTER.labels}")

    if args.shared_answer_prompt:
        from emnlp_evaluation.agents.naive_rag import set_shared_answer_prompt, _ANSWER_PROMPT
        set_shared_answer_prompt(_ANSWER_PROMPT)   # bare, corpus-neutral
        print("[prompt-control] monolithic + scoped systems share one answer prompt")

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
    global _CENTROID_ROUTER, _SOFT_M, _SOFT_GW
    _SOFT_M = args.soft_m
    _SOFT_GW = args.soft_global_weight
    if ("soft_scoped" in args.systems or "composed_" in args.systems) and args.backend == "faiss":
        _CENTROID_ROUTER = CentroidRouter.from_corpus(
            backend._meta, backend._embeddings, spec["scope_field"])
        print(f"[{args.benchmark}] centroid router over "
              f"{len(_CENTROID_ROUTER.scopes)} scopes, m={_SOFT_M}")

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
                    rec = _run_one(system, q["query"], llm, backend, qrec=q)
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
