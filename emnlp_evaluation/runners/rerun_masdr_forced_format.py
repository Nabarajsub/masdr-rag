"""
Forced-format control for the precision-faithfulness paradox (reviewer mFpD).

Takes an existing masdr_rag run, keeps its retrieved evidence exactly as-is,
and re-synthesizes each answer with the *monolithic* answer prompt (single
segment, [Source N] citations, same top-15 cap as run_naive_rag). Any
faithfulness delta vs the original masdr_rag score is then attributable to
answer format, not retrieval or orchestration.

Usage:
    python -m emnlp_evaluation.runners.rerun_masdr_forced_format \\
        --in results/wydot_qwen_bge_m3_baselines.jsonl \\
        --chunk-db judge_assets/wydot_chunks.json \\
        --llm qwen --system masdr_rag
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO.parent))

from emnlp_evaluation.configs import oss_config as cfg
from emnlp_evaluation.llm_providers import get_provider
from emnlp_evaluation.agents.naive_rag import _ANSWER_PROMPT
from emnlp_evaluation.agents.tools_oss import format_chunks


def _dedup_chunks(chunks: list[dict], embedder, thresh: float = 0.95) -> list[dict]:
    """Drop chunks whose embedding cosine vs an already-kept chunk > thresh."""
    import numpy as np
    texts = [c["text"] for c in chunks]
    if len(texts) < 2:
        return chunks
    embs = np.asarray([embedder.embed_query(t[:2000]) for t in texts], dtype=np.float32)
    embs /= (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    kept: list[int] = []
    for i in range(len(chunks)):
        if all(float(embs[i] @ embs[j]) <= thresh for j in kept):
            kept.append(i)
    return [chunks[i] for i in kept]


def _chunks_from_record(rec: dict, db: dict, k: int = 15) -> list[dict]:
    ids = (rec.get("chunk_ids") or [])[:k]
    series = rec.get("chunk_series") or []
    years = rec.get("chunk_years") or []
    sections = rec.get("chunk_sections") or []
    chunks = []
    for i, cid in enumerate(ids):
        chunks.append({
            "id": cid,
            "title": series[i] if i < len(series) else "Unknown",
            "section": sections[i] if i < len(sections) else "",
            "year": years[i] if i < len(years) else "?",
            "text": db.get(str(cid), ""),
        })
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--chunk-db", required=True)
    ap.add_argument("--llm", default="qwen")
    ap.add_argument("--system", default="masdr_rag",
                    help="Which system's records to re-synthesize.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dedup", action="store_true",
                    help="Near-duplicate filter (BGE-M3 cosine > 0.95) on the "
                         "evidence before synthesis (RBJc W4 ablation).")
    args = ap.parse_args()

    db = json.load(open(args.chunk_db))
    llm = get_provider(args.llm)
    embedder = None
    if args.dedup:
        from emnlp_evaluation.embeddings import get_embedder
        embedder = get_embedder("bge_m3")

    in_path = Path(args.inp)
    out_path = Path(args.out) if args.out else (
        cfg.RESULTS_DIR / (in_path.stem + ".forced_format.jsonl"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out_path.exists():
        for line in open(out_path):
            try:
                done.add(json.loads(line)["query_id"])
            except Exception:
                continue

    n = 0
    with open(in_path) as fin, open(out_path, "a") as fout:
        for line in fin:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("system") != args.system or "error" in rec:
                continue
            if rec["query_id"] in done:
                continue
            if args.limit and n >= args.limit:
                break

            chunks = _chunks_from_record(rec, db)
            if args.dedup and embedder is not None:
                chunks = _dedup_chunks(chunks, embedder)
            t0 = time.time()
            res = llm.generate(
                [{"role": "user", "content": _ANSWER_PROMPT.format(
                    context=format_chunks(chunks), query=rec.get("query", ""))}],
                max_new_tokens=1024,
            )
            out = {
                "system": f"{args.system}_forced_format" + ("_dedup" if args.dedup else ""),
                "answer": res.text,
                "n_chunks": len(chunks),
                "chunk_ids": [c["id"] for c in chunks],
                "chunk_series": [c["title"] for c in chunks],
                "chunk_years": [c["year"] for c in chunks],
                "chunk_sections": [c["section"] for c in chunks],
                "prompt_tokens": res.prompt_tokens,
                "completion_tokens": res.completion_tokens,
                "llm_calls": 1,
                "wall_time_s": time.time() - t0,
                "query_id": rec["query_id"],
                "query": rec.get("query", ""),
                "reference_answer": rec.get("reference_answer", ""),
                "category": rec.get("category"),
                "query_type": rec.get("query_type"),
                "llm": rec.get("llm"),
                "embedder": rec.get("embedder"),
                "source_run": in_path.name,
            }
            fout.write(json.dumps(out) + "\n"); fout.flush()
            n += 1
            if n % 20 == 0:
                print(f"[forced-format] {n} rows", flush=True)

    print(f"[forced-format] wrote {n} rows -> {out_path}")


if __name__ == "__main__":
    main()
