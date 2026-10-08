"""
ColBERTv2 baseline runner.

Two systems:
  * colbert_only   — pure ColBERTv2 retrieval (no LLM generation).
  * colbert_qwen   — ColBERTv2 retrieval + Qwen2.5-7B generation.

Run on composite or HotpotQA (CRAG-substitute). Output JSONL matches the
shape latency_pareto / judge expect.
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
from emnlp_evaluation.agents.colbert_backend import load_colbert


BENCHMARK = {
    "composite": {
        "prefix":  _REPO / "emnlp_evaluation" / "composite_corpus" / "data" / "composite",
        "queries": _REPO / "emnlp_evaluation" / "composite_corpus" / "data" / "queries.json",
        "scope_field": "source_type",
        "index_name": "composite_colbert",
    },
    "crag": {
        "prefix":  _REPO / "emnlp_evaluation" / "benchmark" / "crag" / "data" / "crag",
        "queries": _REPO / "emnlp_evaluation" / "benchmark" / "crag" / "data" / "crag.queries.json",
        "scope_field": "source_type",
        "index_name": "crag_colbert",
    },
    "multihop": {
        "prefix":  _REPO / "emnlp_evaluation" / "benchmark" / "multihop_rag" / "data" / "multihop",
        "queries": _REPO / "emnlp_evaluation" / "benchmark" / "multihop_rag" / "data" / "queries.json",
        "scope_field": "source_type",
        "index_name": "multihop_colbert",
    },
    "financebench": {
        "prefix":  _REPO / "emnlp_evaluation" / "benchmark" / "financebench" / "data" / "financebench",
        "queries": _REPO / "emnlp_evaluation" / "benchmark" / "financebench" / "data" / "queries.json",
        "scope_field": "source_type",
        "index_name": "financebench_colbert",
    },
    "mmlu_pro": {
        "prefix":  _REPO / "emnlp_evaluation" / "benchmark" / "mmlu_pro" / "data" / "mmlu_pro",
        "queries": _REPO / "emnlp_evaluation" / "benchmark" / "mmlu_pro" / "data" / "queries.json",
        "scope_field": "source_type",
        "index_name": "mmlu_pro_colbert",
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
    ap.add_argument("--systems", default="colbert_only,colbert_qwen")
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
    backend = load_colbert(spec["prefix"], scope_field=spec["scope_field"],
                            index_name=spec["index_name"])
    print(f"[colbert] backend ready", flush=True)

    with open(spec["queries"]) as f:
        suite = json.load(f)
    if args.limit:
        suite = suite[: args.limit]

    systems = [s.strip() for s in args.systems.split(",")]
    needs_llm = any(s in ("colbert_qwen", "colbert_scoped_qwen") for s in systems)
    llm = get_provider(args.llm) if needs_llm else None

    out_path = Path(args.out) if args.out else (
        cfg.RESULTS_DIR / f"{args.benchmark}_colbert_{'-'.join(systems)}.jsonl"
    )

    # Per-scope wiring: composite has 9 source types -> 9 colbert_<scope>
    # agent names. For colbert_scoped we route by the query's gold scope.
    source_filters = {}
    if args.benchmark == "composite":
        for st in ("gmail","slack","github","jira","confluence","docs",
                   "stackoverflow","helpdesk","reports"):
            source_filters[f"{st}_agent"] = [st]
    backend.source_filters = source_filters

    with open(out_path, "a") as f:
        for q in suite:
            qid = q.get("query_id") or q.get("id")
            query = q.get("query") or q.get("question", "")
            gold_scope = q.get("category") or q.get("gold_source_type") or ""
            for sysn in systems:
                t0 = time.time()
                try:
                    if sysn in ("colbert_scoped", "colbert_scoped_qwen") and gold_scope:
                        # Per-scope ColBERT: filter to the gold scope's index slice
                        chunks = backend.combined_scoped_search(
                            query, f"{gold_scope}_agent")
                    else:
                        chunks = backend.global_search(query)
                    rec: Dict = {
                        "system": sysn, "answer": "",
                        "n_chunks": len(chunks),
                        "chunk_ids": [c["id"] for c in chunks],
                        "chunk_sources": [c["source"] for c in chunks],
                        "prompt_tokens": 0, "completion_tokens": 0,
                        "llm_calls": 0,
                    }
                    if sysn in ("colbert_qwen", "colbert_scoped_qwen") and llm is not None:
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
                        "llm": args.llm if sysn in ("colbert_qwen", "colbert_scoped_qwen") else "none",
                        "embedder": "colbertv2", "benchmark": args.benchmark,
                    })
                except Exception as e:
                    rec = {"query_id": qid, "system": sysn,
                           "error": f"{type(e).__name__}: {e}",
                           "benchmark": args.benchmark}
                f.write(json.dumps(rec) + "\n"); f.flush()
                print(f"[{qid:>16}] [{sysn:<14}] {time.time()-t0:5.1f}s "
                      f"chunks={rec.get('n_chunks','-')}", flush=True)

    print(f"[colbert] wrote {out_path}")


if __name__ == "__main__":
    main()
