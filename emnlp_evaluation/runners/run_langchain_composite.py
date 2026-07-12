"""
LangChain AgentExecutor baseline on composite / HotpotQA — uses the local
FAISS-backed SearchBackend so it never touches Neo4j (and therefore does not
compete with the production client app for connection slots).

Same trace shape as run_langchain_baseline.py so analysis scripts pick it up.
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
from emnlp_evaluation.composite_corpus.local_backend import load_backend
from emnlp_evaluation.agents.langchain_agent import run_langchain
from emnlp_evaluation.runners.run_composite import _build_catalog as _build_composite_catalog
from emnlp_evaluation.runners.run_crag import _build_catalog as _build_crag_catalog


BENCHMARK = {
    "composite": {
        "prefix":  _REPO / "emnlp_evaluation" / "composite_corpus" / "data" / "composite",
        "queries": _REPO / "emnlp_evaluation" / "composite_corpus" / "data" / "queries.json",
        "build_catalog": _build_composite_catalog,
    },
    "crag": {
        "prefix":  _REPO / "emnlp_evaluation" / "benchmark" / "crag" / "data" / "crag",
        "queries": _REPO / "emnlp_evaluation" / "benchmark" / "crag" / "data" / "crag.queries.json",
        "build_catalog": _build_crag_catalog,
    },
}


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
        "iterations": trace.iterations,
        "early_stop_reason": trace.early_stop_reason,
    }
    d.update(extras); return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="composite", choices=list(BENCHMARK))
    ap.add_argument("--llm", default="qwen")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    spec = BENCHMARK[args.benchmark]
    source_filters = spec["build_catalog"]()       # installs catalog into shared modules
    llm = get_provider(args.llm)
    embedder = get_embedder("bge_m3")
    backend = load_backend(spec["prefix"], embedder=embedder,
                            scope_field="source_type", source_filters=source_filters)

    with open(spec["queries"]) as f:
        suite = json.load(f)
    if args.limit:
        suite = suite[: args.limit]

    out_path = Path(args.out) if args.out else (
        cfg.RESULTS_DIR / f"{args.benchmark}_langchain_{args.llm}.jsonl"
    )

    with open(out_path, "a") as f:
        for q in suite:
            qid = q.get("query_id") or q.get("id")
            query = q.get("query") or q.get("question", "")
            t0 = time.time()
            try:
                trace = run_langchain(query, llm=llm, backend=backend)
                rec = _trace_dict(trace, "langchain")
                rec.update({
                    "query_id": qid, "query": query,
                    "reference_answer": q.get("reference_answer", ""),
                    "category": q.get("category") or q.get("gold_source_type"),
                    "llm": args.llm, "embedder": "bge_m3",
                    "benchmark": args.benchmark,
                })
            except Exception as e:
                rec = {"query_id": qid, "system": "langchain",
                       "error": f"{type(e).__name__}: {e}",
                       "benchmark": args.benchmark, "llm": args.llm}
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(f"[{qid:>16}] [langchain] {time.time()-t0:5.1f}s "
                  f"chunks={rec.get('n_chunks','-')} "
                  f"iters={rec.get('iterations','-')}", flush=True)

    print(f"[langchain-{args.benchmark}] wrote {out_path}")


if __name__ == "__main__":
    main()
