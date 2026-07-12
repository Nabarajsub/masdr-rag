"""S-3: HotpotQA-style Exact Match (EM) and token-F1 against gold answers.

The reviewer asked for span-level EM/F1 in addition to Recall@10. This
module re-uses the official HotpotQA / SQuAD normalisation (lowercase,
strip articles, strip punctuation, collapse whitespace) and reports EM
+ F1 per (corpus, system) over the same .judged.jsonl files used
elsewhere in the harness.

Usage:
    python -m emnlp_evaluation.analysis.hotpot_em_f1 \\
        --inputs 'emnlp_evaluation/results/crag_qwen*.judged.jsonl'
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import List


# Official HotpotQA/SQuAD normalisation.
def _normalize(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s or ""))))


def em_score(prediction: str, gold: str) -> int:
    return int(_normalize(prediction) == _normalize(gold))


def f1_score(prediction: str, gold: str) -> float:
    pred_toks = _normalize(prediction).split()
    gold_toks = _normalize(gold).split()
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = Counter(pred_toks) & Counter(gold_toks)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(pred_toks)
    recall = n_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def contains(prediction: str, gold: str) -> int:
    """1 if the normalised gold answer appears as a substring of the
    normalised prediction. The standard ad-hoc evaluation for long-form
    RAG answers against short HotpotQA-style gold spans."""
    return int(_normalize(gold) and _normalize(gold) in _normalize(prediction))


def best_window_em(prediction: str, gold: str, *, max_window: int = 20) -> int:
    """Sliding-window EM: 1 if any contiguous span of <=max_window tokens
    in the prediction exactly matches gold after normalisation."""
    g = _normalize(gold)
    if not g:
        return 0
    g_toks = g.split()
    p_toks = _normalize(prediction).split()
    if len(g_toks) > max_window or len(p_toks) < len(g_toks):
        return int(g in _normalize(prediction))
    n = len(g_toks)
    for i in range(len(p_toks) - n + 1):
        if p_toks[i:i + n] == g_toks:
            return 1
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True,
                    help="Glob of .judged.jsonl files.")
    ap.add_argument("--out", default=None,
                    help="If given, write a JSON summary here.")
    args = ap.parse_args()

    # Multiple .judged.jsonl files in different generations carry the same
    # (corpus, system, query_id) records — dedupe by that key, keeping the
    # first occurrence per (corpus, system, qid).
    seen_qids: dict[tuple[str, str], set] = defaultdict(set)
    by_sys: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"em": [], "f1": [], "win_em": [], "contains": []})
    for path in sorted(glob.glob(args.inputs)):
        for line in open(path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in r:
                continue
            pred = r.get("answer") or ""
            gold = r.get("reference_answer") or ""
            if not gold:
                continue
            sys_name = r.get("system", "?")
            corpus = r.get("benchmark") or "?"
            qid = r.get("query_id")
            key = (corpus, sys_name)
            if qid in seen_qids[key]:
                continue
            seen_qids[key].add(qid)
            by_sys[key]["em"].append(em_score(pred, gold))
            by_sys[key]["f1"].append(f1_score(pred, gold))
            by_sys[key]["win_em"].append(best_window_em(pred, gold))
            by_sys[key]["contains"].append(contains(pred, gold))

    print(f"{'corpus':<10} {'system':<22} {'n':>5} {'EM':>6} {'F1':>6} {'winEM':>6} {'Cont.':>6}")
    summary = {}
    for (corpus, sys_name), d in sorted(by_sys.items()):
        n = len(d["em"])
        em = sum(d["em"]) / max(1, n)
        f1 = sum(d["f1"]) / max(1, n)
        win = sum(d["win_em"]) / max(1, n)
        cont = sum(d["contains"]) / max(1, n)
        print(f"{corpus:<10} {sys_name:<22} {n:>5} {em:>6.3f} {f1:>6.3f} {win:>6.3f} {cont:>6.3f}")
        summary[f"{corpus}/{sys_name}"] = {"n": n, "em": em, "f1": f1,
                                            "win_em": win, "contains": cont}

    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()
