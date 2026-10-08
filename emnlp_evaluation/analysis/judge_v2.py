"""Faithfulness re-judge that shows the judge the evidence the generator saw.

The original judge (judge.py) scores faithfulness as ONE binary verdict
("is EVERY claim supported?") over only the FIRST 15 chunk ids, each cut to
800 chars, with no title headers. Orchestrated systems hand their generator
up to ~190 chunks with title/section/year headers, so claims grounded in
chunk 16+ -- or in a header -- are unverifiable to the judge by construction.

This re-judge (faithfulness only; correctness is carried over unchanged):
  * shows ALL chunks the generator saw (up to --max-chunks), each with its title,
    under a total character budget sized for Qwen-2.5-7B's 32k context; the
    per-chunk allotment shrinks only when the budget binds, and that is recorded
  * scores two things:
      faith_full_binary -- the original all-claims-supported 0/1 question
      faith_claims_frac -- claim-level: judge lists claims, marks each
                           supported/unsupported; fraction supported
                           (the RAGAS-style construct the paper describes)

Input: an existing *.judged.jsonl. Output: <in>.<suffix>.jsonl (default v2judge).
"""
from __future__ import annotations

import argparse, glob, json, re, sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO)); sys.path.insert(0, str(_REPO.parent))
from emnlp_evaluation.llm_providers import get_provider

BINARY = """You are evaluating whether a model-generated answer is grounded in the retrieved sources.

Retrieved sources:
{sources}

Model answer: {answer}

Is EVERY factual claim in the MODEL ANSWER supported by at least one source above? (Citations like [Source 1] don't need to be present — just check whether the claims appear in the sources.)

Reply with a JSON object only:
{{"faithful": 1 or 0, "reason": "<one short sentence>"}}"""

CLAIMS = """You are checking a model-generated answer against the retrieved sources it was given.

Retrieved sources:
{sources}

Model answer: {answer}

Step 1: split the MODEL ANSWER into its individual factual claims (at most 15; skip greetings, hedges, and statements that information is unavailable).
Step 2: for each claim, decide whether at least one source above supports it.

Reply with JSON only, in exactly this form:
{{"claims": [{{"claim": "<short claim>", "supported": 1 or 0}}, ...]}}"""

_SUP = re.compile(r'"supported"\s*:\s*"?([01])')
_FAITH = re.compile(r'"faithful"\s*:\s*"?([01])')


def fmt_sources(rec, db, max_chunks, budget, max_chars):
    ids = (rec.get("chunk_ids") or [])[:max_chunks]
    titles = rec.get("chunk_titles") or []
    per = min(max_chars, max(300, budget // max(1, len(ids))))
    parts, truncated = [], 0
    for i, cid in enumerate(ids):
        body = db.get(str(cid), "(missing)") or ""
        if len(body) > per:
            truncated += 1
        t = titles[i] if i < len(titles) else ""
        parts.append(f"[SOURCE {i+1}: {t}]\n{body[:per]}")
    return "\n---\n".join(parts), len(ids), per, truncated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", required=True)
    ap.add_argument("--chunk-text-db", required=True)
    ap.add_argument("--judge", default="qwen")
    ap.add_argument("--systems", default=None, help="comma list; default all")
    ap.add_argument("--max-chunks", type=int, default=60)
    ap.add_argument("--max-chars", type=int, default=2000)
    ap.add_argument("--budget", type=int, default=60000, help="total source chars")
    ap.add_argument("--suffix", default="v2judge")
    a = ap.parse_args()
    judge = get_provider(a.judge)
    db = json.load(open(a.chunk_text_db))
    keep = set(a.systems.split(",")) if a.systems else None
    for path in glob.glob(a.inputs):
        out = Path(path.replace(".judged.jsonl", f".{a.suffix}.jsonl"))
        seen = set()
        if out.exists():
            seen = {(r["query_id"], r["system"]) for r in map(json.loads, open(out))}
        n = 0
        with open(out, "a") as fo:
            for line in open(path):
                rec = json.loads(line)
                if "error" in rec or (keep and rec.get("system") not in keep):
                    continue
                if (rec["query_id"], rec["system"]) in seen:
                    continue
                src, shown, per, trunc = fmt_sources(rec, db, a.max_chunks, a.budget, a.max_chars)
                ans = rec.get("answer", "")
                b = judge.generate([{"role": "user", "content": BINARY.format(sources=src, answer=ans)}],
                                   max_new_tokens=128, temperature=0.0)
                m = _FAITH.search(b.text or "")
                c = judge.generate([{"role": "user", "content": CLAIMS.format(sources=src, answer=ans)}],
                                   max_new_tokens=1500, temperature=0.0)
                sups = [int(x) for x in _SUP.findall(c.text or "")]
                rec.update({
                    "faith_full_binary": int(m.group(1)) if m else None,
                    "faith_claims_frac": (sum(sups) / len(sups)) if sups else None,
                    "n_claims": len(sups),
                    "judge_chunks_shown": shown, "judge_chars_per_chunk": per,
                    "judge_chunks_truncated": trunc,
                    "n_chunks_generator": len(rec.get("chunk_ids") or []),
                    "answer_chars": len(ans),
                })
                fo.write(json.dumps(rec) + "\n"); fo.flush()
                n += 1
                if n % 25 == 0:
                    print(f"[judge_v2] {path}: {n}", flush=True)
        print(f"[judge_v2] {path} -> {out}")


if __name__ == "__main__":
    main()
