"""
CRAG eval runner — uses the LocalSearchBackend (same as composite).

CRAG categories (finance / sports / music / movie / open) become scoped
domain agents. The orchestrator + ReAct code work unchanged.
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
from emnlp_evaluation.agents.hybrid_routed import run_hybrid_routed, run_regex_scoped
from emnlp_evaluation.agents.naive_rag import run_naive_rag
from emnlp_evaluation.agents.react_baseline import run_react
from emnlp_evaluation.agents.ma_rag import run_ma_rag
from emnlp_evaluation.agents.scout_rag import run_scout_rag


CRAG_PREFIX = str(_REPO / "emnlp_evaluation" / "benchmark" / "crag" / "data" / "crag")
QUERIES_PATH = _REPO / "emnlp_evaluation" / "benchmark" / "crag" / "data" / "crag.queries.json"


# HotpotQA-substitute scoping: 4 alphabetic buckets over Wikipedia article
# titles + an 'other' bucket. Not semantically meaningful, but lets the
# orchestrator exercise routing with a small fixed agent set.
_CRAG_DOMAINS = ["topic_ag", "topic_hm", "topic_ns", "topic_tz", "other"]

_DESC = {
    "search_topic_ag": "Search Wikipedia articles whose title begins A-G.",
    "search_topic_hm": "Search Wikipedia articles whose title begins H-M.",
    "search_topic_ns": "Search Wikipedia articles whose title begins N-S.",
    "search_topic_tz": "Search Wikipedia articles whose title begins T-Z.",
    "search_other":    "Search Wikipedia articles whose title begins with a non-letter.",
    "search_general":  "Search ALL Wikipedia pages (fallback).",
}


def _build_catalog() -> Dict[str, List[str]]:
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
    for d in _CRAG_DOMAINS:
        tname, aname = f"search_{d}", f"{d}_agent"
        tools.append(ToolSpec(name=tname, description=_DESC[tname], parameters=params))
        tool_to_agent[tname] = aname
        source_filters[aname] = [d]
    tools.append(ToolSpec(name="search_general", description=_DESC["search_general"], parameters=params))
    tool_to_agent["search_general"] = "general_agent"

    tc.TOOL_TO_AGENT.clear(); tc.TOOL_TO_AGENT.update(tool_to_agent)
    # orchestrator imports build_tool_catalog at module load; the patched
    # tc.build_tool_catalog never reaches it. The original function reads
    # tc._TOOL_DESCRIPTIONS at call time, so we replace that dict too.
    tc._TOOL_DESCRIPTIONS.clear()
    for t in tools:
        tc._TOOL_DESCRIPTIONS[t.name] = t.description
    tc.build_tool_catalog = lambda: tools
    rb.TOOL_DESCRIPTIONS.clear()
    for t in tools:
        rb.TOOL_DESCRIPTIONS[t.name] = t.description
    # Router config for this corpus: queries never name a title bucket, so no
    # regex rules (the regex tier correctly falls back to global); the LLM
    # router sees the bucket descriptions and can route only when it can infer
    # the answer article's initial.
    from emnlp_evaluation.agents import hybrid_routed as hr
    hr.configure_router(
        regex_rules=[],
        domain_desc=("a Wikipedia corpus partitioned into alphabetic "
                     "article-title buckets (A-G, H-M, N-S, T-Z, other)"),
    )
    return source_filters


def _trace_dict(trace, system: str, **extras):
    d = {
        "system": system, "answer": trace.answer,
        "n_chunks": len(trace.chunks),
        "chunk_ids": [c.get("id") for c in trace.chunks],
        "chunk_sources": [c.get("source") for c in trace.chunks],
        "prompt_tokens": trace.prompt_tokens,
        "completion_tokens": trace.completion_tokens,
        "llm_calls": trace.llm_calls,
        "wall_time_s": trace.wall_time_s,
    }
    d.update(extras)
    return d


def _run_one(system, query, llm, backend):
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
    ap.add_argument("--prefix", default=CRAG_PREFIX)
    ap.add_argument("--queries", default=str(QUERIES_PATH))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--shared-answer-prompt", action="store_true",
                    help="Force shared corpus-neutral answer/system prompts (see naive_rag).")
    args = ap.parse_args()
    if args.shared_answer_prompt:
        from emnlp_evaluation.agents.naive_rag import set_shared_answer_prompt, _ANSWER_PROMPT
        set_shared_answer_prompt(_ANSWER_PROMPT)
        print("[prompt-control] shared corpus-neutral prompts active")

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
        cfg.RESULTS_DIR / f"crag_{args.llm}_{'-'.join(systems)}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: skip (qid, system) pairs already present in the output file.
    done = set()
    if out_path.exists():
        with open(out_path) as fexist:
            for line in fexist:
                try:
                    rec = json.loads(line)
                    if "error" not in rec:
                        done.add((rec.get("query_id"), rec.get("system")))
                except json.JSONDecodeError:
                    continue
        if done:
            print(f"[runner] resume: {len(done)} (qid,system) already present", flush=True)

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
                        "query_type": q.get("question_type"),
                        "llm": args.llm, "embedder": "bge_m3",
                        "benchmark": "crag",
                    })
                except Exception as e:
                    rec = {"query_id": qid, "system": system,
                           "error": f"{type(e).__name__}: {e}",
                           "llm": args.llm, "embedder": "bge_m3",
                           "benchmark": "crag"}
                f.write(json.dumps(rec) + "\n"); f.flush()
                print(f"[{qid:>16}] [{system:<14}] {time.time()-t0:5.1f}s "
                      f"chunks={rec.get('n_chunks','-')}", flush=True)

    print(f"[crag] wrote {out_path}")


if __name__ == "__main__":
    main()
