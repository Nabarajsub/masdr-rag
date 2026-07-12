"""
WYDOT 200-query evaluation runner — provider/embedder agnostic.

Runs the 200-query test suite against any combination of:

  systems:    monolithic, regex_scoped, hybrid_routed, masdr_rag, react, naive
  LLMs:       gemini | qwen | llama
  embedders:  gemini | bge_m3

Output is one JSONL per (system, llm, embedder) tuple under
emnlp_evaluation/results/. Each line records: query, reference answer,
retrieved chunk ids, document_series of each chunk, generated answer,
latency, token usage, routing decision.

Usage:
    python -m emnlp_evaluation.runners.run_wydot_oss \\
        --systems hybrid_routed,masdr_rag,react \\
        --llm qwen \\
        --embedder bge_m3 \\
        [--limit 10] [--query-ids q_001,q_002]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

# Make the repo importable when invoked from anywhere.
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO.parent))

from emnlp_evaluation.configs import oss_config as cfg
from emnlp_evaluation.llm_providers import get_provider
from emnlp_evaluation.embeddings import get_embedder
from emnlp_evaluation.agents.tools_oss import SearchBackend
from emnlp_evaluation.composite_corpus.rerank_backend import RerankBackend
from emnlp_evaluation.agents.orchestrator_oss import run_orchestrator
from emnlp_evaluation.agents.orchestrator_singlecall import run_singlecall_orchestrator
from emnlp_evaluation.agents.hybrid_routed import run_hybrid_routed, run_regex_scoped, run_r2_routed
from emnlp_evaluation.agents.naive_rag import run_naive_rag
from emnlp_evaluation.agents.react_baseline import run_react
from emnlp_evaluation.agents.ma_rag import run_ma_rag
from emnlp_evaluation.agents.scout_rag import run_scout_rag


SUITE_PATH = _REPO / "evaluation" / "test_suite_200.json"


SYSTEMS = {
    "monolithic":    "Monolithic / Naive RAG",
    "regex_scoped":  "Regex + scoped retrieval",
    "hybrid_routed": "Hybrid-Routed (regex → LLM → scoped)",
    "r2_routed":     "R2-Routed (BGE-M3+LogReg trained router → scoped)",
    "masdr_rag":     "MASDR-RAG (orchestrator + multi-tool)",
    "masdr_singlecall": "MASDR-RAG single-synthesis (concat-all-chunks ablation)",
    "react":         "ReAct loop",
    "naive":         "Alias for monolithic (kept for parity)",
    "ma_rag":        "MA-RAG (ported; plan-decompose-execute multi-agent)",
    "scout_rag":     "SCOUT-RAG (reimpl; DRAA/PAGA/OASA/AQAA cooperative agents)",
}


def _resolve_index(embedder_name: str) -> str:
    return cfg.NEO4J_BGE_M3_INDEX if embedder_name == "bge_m3" else cfg.NEO4J_GEMINI_INDEX


def _run_one(system: str, query: str, llm, backend) -> dict:
    if system in ("monolithic", "naive"):
        t = run_naive_rag(query, llm=llm, backend=backend)
        return _trace_dict(t, system, routed_agents=["__global__"])
    if system == "regex_scoped":
        t = run_regex_scoped(query, llm=llm, backend=backend)
        return _trace_dict(t, system,
                           routed_agents=[t.routed_agent] if t.routed_agent else [],
                           route_decision=t.route_decision)
    if system == "hybrid_routed":
        t = run_hybrid_routed(query, llm=llm, backend=backend)
        return _trace_dict(t, system,
                           routed_agents=[t.routed_agent] if t.routed_agent else [],
                           route_decision=t.route_decision)
    if system == "r2_routed":
        t = run_r2_routed(query, llm=llm, backend=backend)
        return _trace_dict(t, system,
                           routed_agents=[t.routed_agent] if t.routed_agent else [],
                           route_decision=t.route_decision)
    if system == "masdr_rag":
        t = run_orchestrator(query, llm=llm, backend=backend)
        return _trace_dict(t, system, routed_agents=t.routed_agents,
                           tool_calls=t.tool_calls)
    if system == "masdr_singlecall":
        t = run_singlecall_orchestrator(query, llm=llm, backend=backend)
        return _trace_dict(t, system, routed_agents=t.routed_agents,
                           tool_calls=t.tool_calls)
    if system == "react":
        t = run_react(query, llm=llm, backend=backend)
        return _trace_dict(t, system, routed_agents=t.routed_agents,
                           tool_calls=t.actions, iterations=t.iterations,
                           early_stop_reason=t.early_stop_reason)
    if system == "ma_rag":
        t = run_ma_rag(query, llm=llm, backend=backend)
        return _trace_dict(t, system, routed_agents=["__global__"],
                           plan=t.plan, step_outputs=t.step_outputs)
    if system == "scout_rag":
        t = run_scout_rag(query, llm=llm, backend=backend)
        return _trace_dict(t, system, routed_agents=t.routed_agents,
                           tiers=t.tiers, strategies=t.strategies,
                           quality_history=t.quality_history,
                           iterations=t.iterations,
                           early_stop_reason=t.early_stop_reason)
    raise ValueError(f"Unknown system: {system}")


def _trace_dict(trace, system: str, **extras) -> dict:
    d = {
        "system": system,
        "answer": trace.answer,
        "n_chunks": len(trace.chunks),
        "chunk_ids": [c.get("id") for c in trace.chunks],
        "chunk_series": [c.get("title") or c.get("source") for c in trace.chunks],
        "chunk_years": [c.get("year") for c in trace.chunks],
        "chunk_sections": [c.get("section") for c in trace.chunks],
        "prompt_tokens": trace.prompt_tokens,
        "completion_tokens": trace.completion_tokens,
        "llm_calls": trace.llm_calls,
        "wall_time_s": trace.wall_time_s,
    }
    d.update(extras)
    return d


def _load_suite() -> List[Dict]:
    with open(SUITE_PATH) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="hybrid_routed,masdr_rag,react",
                    help="Comma-separated subset of: " + ",".join(SYSTEMS))
    ap.add_argument("--llm", default="qwen",
                    help="LLM backend: gemini/qwen/llama or an OpenRouter slug "
                         "prefixed by 'openrouter:' (e.g. openrouter:anthropic/claude-sonnet-4.5).")
    ap.add_argument("--embedder", default="bge_m3", choices=cfg.SUPPORTED_EMBEDDERS)
    ap.add_argument("--limit", type=int, default=None,
                    help="Run only the first N queries (smoke test).")
    ap.add_argument("--query-ids", default=None,
                    help="Comma-separated query ids (e.g. q_001,q_002).")
    ap.add_argument("--shard", type=int, default=None,
                    help="Shard index (0-based) for parallel runs.")
    ap.add_argument("--of", type=int, default=None,
                    help="Total number of shards. Combined with --shard.")
    ap.add_argument("--out", default=None,
                    help="Output JSONL path (default: results/wydot_<llm>_<embed>_<systems>.jsonl).")
    ap.add_argument("--resume", action="store_true",
                    help="Skip queries already present in the output file.")
    ap.add_argument("--rerank", action="store_true",
                    help="Wrap backend with BGE-reranker-v2-m3 (top-30 -> top-10).")
    args = ap.parse_args()

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    unknown = [s for s in systems if s not in SYSTEMS]
    if unknown:
        raise SystemExit(f"Unknown systems: {unknown}")

    print(f"[runner] llm={args.llm} embedder={args.embedder} systems={systems}", flush=True)
    if args.llm.startswith("openrouter:"):
        llm = get_provider("openrouter", model_name=args.llm.split(":", 1)[1])
    else:
        llm = get_provider(args.llm)
    embedder = get_embedder(args.embedder)
    backend = SearchBackend(
        embedder=embedder, vector_index=_resolve_index(args.embedder)
    )
    if args.rerank:
        backend = RerankBackend(backend)

    suite = _load_suite()
    if args.query_ids:
        wanted = {q.strip() for q in args.query_ids.split(",")}
        suite = [q for q in suite if q["id"] in wanted]
    if args.shard is not None and args.of:
        suite = [q for i, q in enumerate(suite) if i % args.of == args.shard]
        print(f"[runner] shard {args.shard}/{args.of}: {len(suite)} queries")
    if args.limit:
        suite = suite[: args.limit]

    shard_tag = f"_shard{args.shard}of{args.of}" if args.shard is not None else ""
    rerank_tag = "_rerank" if args.rerank else ""
    llm_tag = args.llm.replace("openrouter:", "or_").replace("/", "-")
    out_path = Path(args.out) if args.out else (
        cfg.RESULTS_DIR / f"wydot_{llm_tag}_{args.embedder}{rerank_tag}_{'-'.join(systems)}{shard_tag}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if args.resume and out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done.add((rec["query_id"], rec["system"]))
                except json.JSONDecodeError:
                    continue
        print(f"[runner] resuming, {len(done)} records already present", flush=True)

    with open(out_path, "a") as f:
        for q in suite:
            for system in systems:
                if (q["id"], system) in done:
                    continue
                t0 = time.time()
                try:
                    rec = _run_one(system, q["query"], llm, backend)
                    rec.update({
                        "query_id": q["id"],
                        "query": q["query"],
                        "reference_answer": q.get("reference_answer", ""),
                        "category": q.get("category"),
                        "relevant_section": q.get("relevant_section"),
                        "relevant_title": q.get("relevant_title"),
                        "query_type": q.get("query_type"),
                        "llm": args.llm,
                        "embedder": args.embedder,
                    })
                except Exception as e:
                    rec = {
                        "query_id": q["id"], "system": system,
                        "error": f"{type(e).__name__}: {e}",
                        "llm": args.llm, "embedder": args.embedder,
                    }
                f.write(json.dumps(rec) + "\n"); f.flush()
                print(f"[{q['id']:>6}] [{system:<14}] {time.time()-t0:5.1f}s "
                      f"chunks={rec.get('n_chunks','-')} "
                      f"tokens={rec.get('prompt_tokens',0)+rec.get('completion_tokens',0):>6}",
                      flush=True)

    print(f"[runner] wrote {out_path}")


if __name__ == "__main__":
    main()
