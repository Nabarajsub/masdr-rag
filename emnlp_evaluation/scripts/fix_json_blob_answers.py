"""Repair MA-RAG / SCOUT-RAG result files whose ``answer`` field is a
JSON-shaped blob (e.g.\\ ``'{"output": "...", "answer": "", "score": 0}'``).

The bug: when the synthesis LLM emitted valid JSON with ``"answer": ""``,
our ``_answer_or_raw`` fallback returned the raw model text — which IS
that same JSON string. The judge would mark those answers wrong on
appearance alone, unfairly tanking the baseline.

This script post-processes existing JSONL files. For each record whose
``answer`` field starts with ``{`` and contains ``"answer"``, we extract
a real prose answer using this fallback chain:

  1. Inner JSON ``answer`` field if non-empty.
  2. For MA-RAG: the last ``step_outputs`` step's ``answer``, then its
     ``analysis``.
  3. Failing all that, an explicit refusal string so the judge sees
     a clean "I cannot determine" rather than JSON syntax.

Outputs go to ``<original>.fixed.jsonl`` next to the input file. We
never overwrite the original.

Usage:
    python -m emnlp_evaluation.scripts.fix_json_blob_answers \\
        --files <path1.jsonl> <path2.jsonl> ...
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict


_JSON_BLOB_HEAD = re.compile(r'^\s*\{[^{]*"answer"', re.DOTALL)
_REFUSAL = "I cannot determine the answer from the provided context."


def _as_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return " ".join(_as_text(x) for x in v)
    if isinstance(v, dict):
        return " ".join(_as_text(x) for x in v.values())
    return str(v)


def _looks_like_blob(ans: str) -> bool:
    s = (ans or "").strip()
    return bool(s) and s.startswith("{") and '"answer"' in s


def _try_parse_inner(ans: str) -> Dict[str, Any] | None:
    s = (ans or "").strip()
    # Strip code fence if present.
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, re.DOTALL)
    cand = m.group(1) if m else s
    # Trim anything after the last brace if the model trailed off.
    last = cand.rfind("}")
    if last > 0:
        cand = cand[: last + 1]
    try:
        d = json.loads(cand)
        return d if isinstance(d, dict) else None
    except json.JSONDecodeError:
        try:
            d = json.loads(cand.replace("'", '"'))
            return d if isinstance(d, dict) else None
        except json.JSONDecodeError:
            return None


def _fix_record(rec: Dict[str, Any]) -> tuple[Dict[str, Any], bool]:
    """Return (possibly-edited record, was_fixed)."""
    a = rec.get("answer")
    if not isinstance(a, str) or not _looks_like_blob(a):
        return rec, False

    # 1. Try the inner JSON's answer field.
    inner = _try_parse_inner(a)
    if inner:
        cand = _as_text(inner.get("answer", "")).strip()
        if cand and not _looks_like_blob(cand):
            rec = dict(rec); rec["answer"] = cand
            rec["_postproc"] = "inner_json_answer"
            return rec, True

    # 2. MA-RAG: walk step_outputs in reverse for the most recent prose.
    sos = rec.get("step_outputs") or []
    if isinstance(sos, list):
        for so in reversed(sos):
            if not isinstance(so, dict):
                continue
            for fld in ("answer", "analysis"):
                cand = _as_text(so.get(fld, "")).strip()
                if cand and not _looks_like_blob(cand):
                    rec = dict(rec); rec["answer"] = cand
                    rec["_postproc"] = f"step_outputs[-].{fld}"
                    return rec, True

    # 3. Last resort: explicit refusal, so the judge sees prose.
    rec = dict(rec); rec["answer"] = _REFUSAL
    rec["_postproc"] = "refusal_fallback"
    return rec, True


def fix_file(in_path: Path, out_path: Path) -> tuple[int, int]:
    n = 0; fixed = 0
    with open(in_path) as fi, open(out_path, "w") as fo:
        for line in fi:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                fo.write(line + "\n"); continue
            n += 1
            new_rec, was_fixed = _fix_record(rec)
            if was_fixed:
                fixed += 1
            fo.write(json.dumps(new_rec) + "\n")
    return n, fixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True,
                    help="JSONL files to repair")
    args = ap.parse_args()

    summary = []
    for fp in args.files:
        in_path = Path(fp)
        if not in_path.exists():
            print(f"[skip] {in_path}: not found", file=sys.stderr); continue
        out_path = in_path.with_suffix(in_path.suffix + ".fixed.jsonl") \
            if not in_path.name.endswith(".fixed.jsonl") \
            else in_path
        n, fixed = fix_file(in_path, out_path)
        pct = 100 * fixed / n if n else 0
        print(f"  {in_path.name}: {n} records, {fixed} repaired ({pct:.1f}%) -> {out_path.name}")
        summary.append((in_path.name, n, fixed))

    print()
    print("=== summary ===")
    for name, n, fixed in summary:
        pct = 100 * fixed / n if n else 0
        print(f"  {name:50s} {fixed:4d}/{n:4d} repaired ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
