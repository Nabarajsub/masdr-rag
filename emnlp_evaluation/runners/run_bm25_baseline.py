"""
BM25 baseline runner.

Two systems:
  * bm25_only       — pure BM25 retrieval (top-k chunks, NO LLM generation).
                      Reported for retrieval-precision comparison.
  * bm25_qwen       — BM25 retrieval + Qwen2.5-7B answer synthesis.
                      Direct apples-to-apples vs. dense retrievers.

Defaults to the composite corpus + composite query set. Use --benchmark crag
to run on HotpotQA. Use --benchmark wydot to run against the WYDOT chunks
(requires a parquet dump of WYDOT — see TODO at the bottom).
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
from emnlp_evaluation.agents.bm25_backend import load_bm25
from emnlp_evaluation.agents.naive_rag import run_naive_rag


BENCHMARK = {
    "composite": {
        "prefix":  _REPO / "emnlp_evaluation" / "composite_corpus" / "data" / "composite",
        "queries": _REPO / "emnlp_evaluation" / "composite_corpus" / "data" / "queries.json",
        "scope_field": "source_type",
    },
    "crag": {
        "prefix":  _REPO / "emnlp_evaluation" / "benchmark" / "crag" / "data" / "crag",
        "queries": _REPO / "emnlp_evaluation" / "benchmark" / "crag" / "data" / "crag.queries.json",
        "scope_field": "source_type",
    },
}


def _ans_prompt(query: str, chunks: List[Dict]) -> str:
    ctx = "\n---\n".join(
        f"[SOURCE {i+1}: {c.get('title','')}, src={c.get('source','')}]\n{c['text']}"
        for i, c in enumerate(chunks)
    ) or "(no results)"
    return (f"Answer the question using ONLY the retrieved sources. "
            f"Cite as [Source 1], [Source 2], etc.\n\n"
            f"Retrieved sources:\n{ctx}\n\nQuestion: {query}\nAnswer:")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="composite", choices=list(BENCHMARK))
    ap.add_argument("--systems", default="bm25_only,bm25_qwen")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--llm", default="qwen")
    ap.add_argument("--out", default=None)
    ap.add_argument("--shared-answer-prompt", action="store_true",
                    help="Force the shared corpus-neutral answer prompt (see naive_rag).")
    args = ap.parse_args()
    if args.shared_answer_prompt:
        from emnlp_evaluation.agents.naive_rag import set_shared_answer_prompt, _ANSWER_PROMPT
        set_shared_answer_prompt(_ANSWER_PROMPT)
        print("[prompt-control] shared corpus-neutral answer prompt active")

    spec = BENCHMARK[args.benchmark]
    backend = load_bm25(spec["prefix"], scope_field=spec["scope_field"])
    print(f"[bm25] loaded {len(backend._meta)} chunks for benchmark={args.benchmark}", flush=True)

    with open(spec["queries"]) as f:
        suite = json.load(f)
    if args.limit:
        suite = suite[: args.limit]

    systems = [s.strip() for s in args.systems.split(",")]
    needs_llm = "bm25_qwen" in systems
    llm = get_provider(args.llm) if needs_llm else None

    out_path = Path(args.out) if args.out else (
        cfg.RESULTS_DIR / f"{args.benchmark}_bm25_{'-'.join(systems)}.jsonl"
    )

    with open(out_path, "a") as f:
        for q in suite:
            qid = q.get("query_id") or q.get("id")
            query = q.get("query") or q.get("question", "")
            for sysn in systems:
                t0 = time.time()
                try:
                    chunks = backend.global_search(query)
                    rec: Dict = {
                        "system": sysn, "answer": "",
                        "n_chunks": len(chunks),
                        "chunk_ids": [c["id"] for c in chunks],
                        "chunk_sources": [c["source"] for c in chunks],
                        "prompt_tokens": 0, "completion_tokens": 0,
                        "llm_calls": 0,
                    }
                    if sysn == "bm25_qwen" and llm is not None:
                        prompt = _ans_prompt(query, chunks)
                        res = llm.generate(
                            [{"role": "user", "content": prompt}],
                            max_new_tokens=512,
                        )
                        rec["answer"] = res.text
                        rec["prompt_tokens"] = res.prompt_tokens
                        rec["completion_tokens"] = res.completion_tokens
                        rec["llm_calls"] = 1
                    rec["wall_time_s"] = time.time() - t0
                    rec.update({
                        "query_id": qid, "query": query,
                        "reference_answer": q.get("reference_answer", ""),
                        "category": q.get("category") or q.get("gold_source_type"),
                        "llm": args.llm if sysn == "bm25_qwen" else "none",
                        "embedder": "bm25", "benchmark": args.benchmark,
                    })
                except Exception as e:
                    rec = {"query_id": qid, "system": sysn,
                           "error": f"{type(e).__name__}: {e}",
                           "benchmark": args.benchmark}
                f.write(json.dumps(rec) + "\n"); f.flush()
                print(f"[{qid:>16}] [{sysn:<12}] {time.time()-t0:5.1f}s "
                      f"chunks={rec.get('n_chunks','-')}", flush=True)

    print(f"[bm25] wrote {out_path}")


if __name__ == "__main__":
    main()
