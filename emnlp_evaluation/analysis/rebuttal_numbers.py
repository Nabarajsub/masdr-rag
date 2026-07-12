"""Aggregate every rebuttal experiment into one report.

Usage:
    python -m emnlp_evaluation.analysis.rebuttal_numbers
"""
from __future__ import annotations

import glob
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"


def _stats(path, system_filter=None):
    out = defaultdict(lambda: {"n": 0, "corr": 0, "faith": [], "p50": []})
    if not Path(path).exists():
        return out
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "error" in r:
            continue
        s = r.get("system", "?")
        if system_filter and s not in system_filter:
            continue
        st = out[s]
        st["n"] += 1
        st["corr"] += r.get("correctness", 0) or 0
        if isinstance(r.get("faithfulness"), (int, float)):
            st["faith"].append(r["faithfulness"])
        if isinstance(r.get("wall_time_s"), (int, float)):
            st["p50"].append(r["wall_time_s"])
    return out


def _fmt(st):
    import statistics
    f = sum(st["faith"]) / len(st["faith"]) if st["faith"] else float("nan")
    p50 = statistics.median(st["p50"]) if st["p50"] else float("nan")
    return f"n={st['n']:4d}  corr={st['corr']/max(st['n'],1):.3f}  faith={f:.3f}  p50={p50:.2f}s"


def section(title):
    print(f"\n{'='*72}\n{title}\n{'='*72}")


def main():
    # ---- E1: fixed-router cross-corpus re-runs -------------------------
    for corpus in ("composite", "multihop", "mmlu_pro"):
        f = RESULTS / f"{corpus}_qwen_fixedrouter_monolithic-regex_scoped-hybrid_routed-masdr_rag.judged.jsonl"
        section(f"E1 {corpus} (fixed router)")
        for sysname, st in sorted(_stats(f).items()):
            print(f"  {sysname:22s} {_fmt(st)}")

    # ---- E2: confusion diagonals on fixed composite run ----------------
    section("E2 confusion diagonals (composite, fixed router)")
    f = RESULTS / "composite_qwen_fixedrouter_monolithic-regex_scoped-hybrid_routed-masdr_rag.jsonl"
    if not f.exists():
        f = Path(str(f).replace(".jsonl", ".judged.jsonl"))
    SOURCES = ["helpdesk", "confluence", "stackoverflow", "gmail", "docs"]
    ALL9 = ["gmail", "slack", "github", "jira", "confluence",
            "docs", "stackoverflow", "helpdesk", "reports"]
    if f.exists():
        recs = [json.loads(l) for l in open(f)]
        for label, srcs in (("fig5-5src", SOURCES), ("all-9src", ALL9)):
            for sysname in ("monolithic", "regex_scoped", "hybrid_routed", "masdr_rag"):
                by_gold = defaultdict(Counter)
                for r in recs:
                    if r.get("system") != sysname or "error" in r:
                        continue
                    g = r.get("category")
                    if g not in srcs:
                        continue
                    for s in (r.get("chunk_sources") or []):
                        if s in srcs:
                            by_gold[g][s] += 1
                ds = [by_gold[g][g] / sum(by_gold[g].values())
                      for g in srcs if sum(by_gold[g].values())]
                if ds:
                    print(f"  [{label}] {sysname:15s} diag avg = {sum(ds)/len(ds):.3f}  ({len(ds)} gold sources)")

    # ---- routing behaviour on the fixed run -----------------------------
    section("E1 routing distribution (composite hybrid_routed)")
    if f.exists():
        hr = [r for r in recs if r.get("system") == "hybrid_routed" and "error" not in r]
        c = Counter(r.get("routed_agent") for r in hr)
        gold_ok = sum(1 for r in hr if str(r.get("routed_agent", "")).startswith(str(r.get("category"))))
        print(f"  routed: {dict(c)}")
        print(f"  routed to gold-source agent: {gold_ok}/{len(hr)}")

    # ---- E3: forced format ----------------------------------------------
    section("E3 forced-format control (WYDOT)")
    pairs = [
        ("BGE-M3 stack   ", RESULTS / "wydot_qwen_bge_m3_masdr_forced_format.judged.jsonl",
         "orig masdr faith=.391 corr=.274 (Tables 9/10)"),
        ("Gemini-emb stack", RESULTS / "wydot_qwen_gemini_masdr_forced_format.judged.jsonl",
         "orig masdr faith=.467 corr=.289 | mono faith=.596 corr=.343"),
    ]
    for label, fp, ref in pairs:
        for sysname, st in sorted(_stats(fp).items()):
            print(f"  {label} {sysname:28s} {_fmt(st)}   [{ref}]")

    # ---- dedup ablation --------------------------------------------------
    section("Dedup ablation (RBJc W4)")
    for label, fp in (("BGE-M3 stack   ", RESULTS / "wydot_qwen_bge_m3_masdr_ff_dedup.judged.jsonl"),
                      ("Gemini-emb stack", RESULTS / "wydot_qwen_gemini_masdr_ff_dedup.judged.jsonl")):
        for sysname, st in sorted(_stats(fp).items()):
            print(f"  {label} {sysname:28s} {_fmt(st)}")

    # ---- ext baselines re-judged with chunk DB ---------------------------
    section("MA-RAG / SCOUT-RAG protocol-equalized re-judge")
    for corpus in ("wydot", "composite", "multihop"):
        fp = RESULTS / f"{corpus}_qwen_extbase_rejudge.judged.jsonl"
        for sysname, st in sorted(_stats(fp).items()):
            print(f"  {corpus:10s} {sysname:12s} {_fmt(st)}")

    # ---- fallback sweep ---------------------------------------------------
    section("R2 confidence-fallback sweep (precomputed)")
    fp = RESULTS / "fallback_sweep.json"
    if fp.exists():
        for row in json.load(open(fp))["taus"]:
            print(f"  tau={row['tau']:4.2f} fallback={row['fallback_rate']:5.1%} "
                  f"corr={row['correctness']:.3f} faith={row['faithfulness']:.3f}")


if __name__ == "__main__":
    main()
