"""
LangChain AgentExecutor baseline runner.

Same arguments as run_wydot_oss.py but only runs the `langchain` system.
Output JSONL is in the same shape so latency_pareto / judge pick it up.
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
from emnlp_evaluation.llm_providers import get_provider
from emnlp_evaluation.embeddings import get_embedder
from emnlp_evaluation.agents.tools_oss import SearchBackend
from emnlp_evaluation.agents.langchain_agent import run_langchain


SUITE_PATH = _REPO / "evaluation" / "test_suite_200.json"


def _resolve_index(embedder_name: str) -> str:
    return cfg.NEO4J_BGE_M3_INDEX if embedder_name == "bge_m3" else cfg.NEO4J_GEMINI_INDEX


def _trace_dict(trace, system: str, **extras) -> dict:
    d = {
        "system": system, "answer": trace.answer,
        "n_chunks": len(trace.chunks),
        "chunk_ids": [c.get("id") for c in trace.chunks],
        "chunk_series": [c.get("title") for c in trace.chunks],
        "prompt_tokens": trace.prompt_tokens,
        "completion_tokens": trace.completion_tokens,
        "llm_calls": trace.llm_calls,
        "wall_time_s": trace.wall_time_s,
        "iterations": trace.iterations,
        "early_stop_reason": trace.early_stop_reason,
    }
    d.update(extras); return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="qwen", choices=cfg.SUPPORTED_PROVIDERS)
    ap.add_argument("--embedder", default="gemini", choices=cfg.SUPPORTED_EMBEDDERS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--of", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    llm = get_provider(args.llm)
    embedder = get_embedder(args.embedder)
    backend = SearchBackend(embedder=embedder, vector_index=_resolve_index(args.embedder))

    with open(SUITE_PATH) as f:
        suite = json.load(f)
    if args.shard is not None and args.of:
        suite = [q for i, q in enumerate(suite) if i % args.of == args.shard]
    if args.limit:
        suite = suite[: args.limit]

    shard_tag = f"_shard{args.shard}of{args.of}" if args.shard is not None else ""
    out_path = Path(args.out) if args.out else (
        cfg.RESULTS_DIR / f"wydot_{args.llm}_{args.embedder}_langchain{shard_tag}.jsonl"
    )

    with open(out_path, "a") as f:
        for q in suite:
            qid = q["id"]; t0 = time.time()
            try:
                trace = run_langchain(q["query"], llm=llm, backend=backend)
                rec = _trace_dict(trace, "langchain")
                rec.update({
                    "query_id": qid, "query": q["query"],
                    "reference_answer": q.get("reference_answer", ""),
                    "category": q.get("category"),
                    "query_type": q.get("query_type"),
                    "llm": args.llm, "embedder": args.embedder,
                })
            except Exception as e:
                rec = {"query_id": qid, "system": "langchain",
                       "error": f"{type(e).__name__}: {e}",
                       "llm": args.llm, "embedder": args.embedder}
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(f"[{qid:>6}] [langchain]  {time.time()-t0:5.1f}s "
                  f"chunks={rec.get('n_chunks','-')} iters={rec.get('iterations','-')} "
                  f"tokens={rec.get('prompt_tokens',0)+rec.get('completion_tokens',0):>6}",
                  flush=True)

    print(f"[langchain] wrote {out_path}")


if __name__ == "__main__":
    main()
