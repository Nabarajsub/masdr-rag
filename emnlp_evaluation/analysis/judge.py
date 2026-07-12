"""
LLM-as-judge for correctness and faithfulness.

Two metrics per row:
  * correctness  — does the generated answer agree with the reference?
  * faithfulness — is every claim in the generated answer grounded in the
                   retrieved chunks?

Both are scored 0/1 by a judge LLM (Qwen by default, swappable to Gemini).
Output: a sibling .jsonl with the same query_id/system rows plus
`correctness`, `faithfulness`, and a 1-sentence rationale.

Usage:
    python -m emnlp_evaluation.analysis.judge \\
        --in results/wydot_qwen_bge_m3_*.jsonl \\
        --judge qwen
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO.parent))

from emnlp_evaluation.llm_providers import get_provider


_CORRECTNESS_PROMPT = """You are evaluating a model-generated answer against a reference answer.

Question: {query}
Reference answer: {reference}
Model answer: {answer}

Does the MODEL ANSWER convey the same factual content as the REFERENCE ANSWER, allowing for paraphrasing? Reply with a JSON object only:
{{"correct": 1 or 0, "reason": "<one short sentence>"}}"""


_FAITHFULNESS_PROMPT = """You are evaluating whether a model-generated answer is grounded in the retrieved sources.

Retrieved sources:
{sources}

Model answer: {answer}

Is EVERY factual claim in the MODEL ANSWER supported by at least one source above? (Citations like [Source 1] don't need to be present — just check whether the claims appear in the sources.)

Reply with a JSON object only:
{{"faithful": 1 or 0, "reason": "<one short sentence>"}}"""


_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_json(text: str, key: str, default=0):
    m = _JSON_RE.search(text)
    if not m:
        return {key: default, "reason": "judge returned no JSON"}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {key: default, "reason": "json parse error"}
    if key not in obj:
        obj[key] = default
    return obj


def _format_sources(chunks: List[Dict] | List[str], texts_lookup=None,
                    max_chars_per_chunk: int = 800) -> str:
    """Render chunks for the faithfulness prompt. We truncate each chunk to
    `max_chars_per_chunk` so the prompt fits in a 7B model's context window
    even on backbones like Claude that retrieve many long chunks."""
    if texts_lookup is None:
        return "(chunk texts unavailable)"
    parts = []
    for i, cid in enumerate(chunks):
        body = (texts_lookup.get(cid, "(missing)") or "")[:max_chars_per_chunk]
        parts.append(f"[SOURCE {i+1}]\n{body}")
    return "\n---\n".join(parts)


def judge_file(in_path: Path, judge, *, chunk_text_db=None, force: bool=False) -> Path:
    out_path = in_path.with_suffix(".judged.jsonl")
    seen = set()
    if out_path.exists() and not force:
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line); seen.add((r["query_id"], r["system"]))
                except Exception:
                    continue
    elif force and out_path.exists():
        out_path.unlink()

    n = 0
    with open(in_path) as fin, open(out_path, "a") as fout:
        for line in fin:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (rec.get("query_id"), rec.get("system"))
            if key in seen or "error" in rec:
                continue

            # Correctness
            corr_prompt = _CORRECTNESS_PROMPT.format(
                query=rec.get("query", ""),
                reference=rec.get("reference_answer", ""),
                answer=rec.get("answer", ""),
            )
            cres = judge.generate([{"role": "user", "content": corr_prompt}],
                                  max_new_tokens=128, temperature=0.0)
            corr = _parse_json(cres.text, "correct")

            # Faithfulness (only if we have chunk texts available).
            # Truncate to top-15 chunks so the prompt fits in Qwen-7B's
            # context window even on systems like MASDR-RAG that retrieve
            # ~34 chunks across multiple tool calls.
            chunk_ids = (rec.get("chunk_ids") or [])[:15]
            sources = _format_sources(chunk_ids, chunk_text_db)
            faith_prompt = _FAITHFULNESS_PROMPT.format(
                sources=sources, answer=rec.get("answer", ""),
            )
            fres = judge.generate([{"role": "user", "content": faith_prompt}],
                                  max_new_tokens=128, temperature=0.0)
            faith = _parse_json(fres.text, "faithful")

            rec["correctness"] = int(corr.get("correct", 0))
            rec["correctness_reason"] = corr.get("reason", "")
            rec["faithfulness"] = int(faith.get("faithful", 0))
            rec["faithfulness_reason"] = faith.get("reason", "")
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            n += 1
            if n % 20 == 0:
                print(f"[judge] {n} rows judged", flush=True)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", required=True,
                    help="Glob of input JSONL files (quote it).")
    ap.add_argument("--judge", default="qwen", choices=("qwen", "llama", "gemini"))
    ap.add_argument("--chunk-text-db", default=None,
                    help="Optional JSON mapping chunk_id->text for faithfulness scoring.")
    ap.add_argument("--force", action="store_true",
                    help="Re-judge even if .judged.jsonl already exists; truncates the output.")
    args = ap.parse_args()

    judge = get_provider(args.judge)
    chunk_db = None
    if args.chunk_text_db:
        with open(args.chunk_text_db) as f:
            chunk_db = json.load(f)

    for path in glob.glob(args.inputs):
        out = judge_file(Path(path), judge, chunk_text_db=chunk_db, force=args.force)
        print(f"[judge] {path} -> {out}")


if __name__ == "__main__":
    main()
