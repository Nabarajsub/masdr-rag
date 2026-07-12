"""Aggregate the paradox-cause (S-2) and cross-corpus numbers for the paper.

Reads every relevant .judged.jsonl, computes faithfulness / correctness /
Recall@10 per (corpus, system), and prints LaTeX-ready tables that can be
pasted into main.tex.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

RESULTS = Path("emnlp_evaluation/results")


def stats(path: Path, *, filter_system: str | None = None):
    if not path.exists():
        return None
    n = 0; faith = []; corr = []; hit = []; chunks = []; calls = []; ptok = []; ctok = []
    for line in path.open():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "error" in r:
            continue
        if filter_system and r.get("system") != filter_system:
            continue
        n += 1
        if r.get("faithfulness") is not None:
            faith.append(float(r["faithfulness"]))
        if r.get("correctness") is not None:
            corr.append(float(r["correctness"]))
        # Composite/CRAG: chunk-level gold (gold_chunk_id vs chunk_ids).
        gold_chunk = r.get("gold_chunk_id")
        if gold_chunk:
            hit.append(1 if gold_chunk in (r.get("chunk_ids") or [])[:10] else 0)
        # WYDOT: title-level gold (relevant_title vs chunk_series).
        rel_title = r.get("relevant_title")
        if rel_title and not gold_chunk:
            titles = [(t or "").strip().lower() for t in (r.get("chunk_series") or [])[:10]]
            hit.append(1 if rel_title.strip().lower() in titles else 0)
        chunks.append(r.get("n_chunks", 0))
        calls.append(r.get("llm_calls", 0))
        ptok.append(r.get("prompt_tokens", 0))
        ctok.append(r.get("completion_tokens", 0))
    avg = lambda L: sum(L) / max(1, len(L)) if L else 0.0
    return dict(n=n,
                faith=avg(faith), corr=avg(corr), recall_at_10=avg(hit),
                chunks=avg(chunks), calls=avg(calls),
                tokens=avg(ptok) + avg(ctok))


def print_table(rows: Iterable[tuple[str, dict | None]], title: str):
    print(f"\n== {title} ==")
    print(f"{'system':<25} {'n':>4} {'Faith':>6} {'Corr':>6} {'R@10':>6} {'chunks':>6} {'calls':>5} {'tok':>6}")
    for name, s in rows:
        if s is None or s["n"] == 0:
            print(f"{name:<25} {'-':>4} {'-':>6} {'-':>6} {'-':>6} {'-':>6} {'-':>5} {'-':>6}  (no data)")
            continue
        print(f"{name:<25} {s['n']:>4} {s['faith']:>6.3f} {s['corr']:>6.3f} "
              f"{s['recall_at_10']:>6.3f} {s['chunks']:>6.1f} {s['calls']:>5.2f} {s['tokens']:>6.0f}")


def latex_singlecall(rows: list[tuple[str, dict | None]], corpus: str) -> str:
    """One row block for tab:singlecall, one corpus."""
    out = []
    out.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{corpus}}}}} \\\\")
    for name, s in rows:
        if s is None or s["n"] == 0:
            continue
        out.append(f"\\quad {name} & ${s['recall_at_10']:.3f}$ & ${s['faith']:.3f}$ & ${s['corr']:.3f}$ \\\\")
    return "\n".join(out)


def main():
    # composite
    comp_base = RESULTS / "composite_qwen_monolithic-regex_scoped-hybrid_routed-masdr_rag-react.judged.jsonl"
    comp_rows = [
        ("Monolithic",        stats(comp_base, filter_system="monolithic")),
        ("Regex-scoped",      stats(comp_base, filter_system="regex_scoped")),
        ("Hybrid-Routed",     stats(comp_base, filter_system="hybrid_routed")),
        ("MASDR-RAG",         stats(RESULTS / "composite_qwen_masdr_rag.judged.jsonl")),
        ("MASDR-SingleCall",  stats(RESULTS / "composite_qwen_masdr_singlecall.judged.jsonl")),
        ("ReAct",             stats(comp_base, filter_system="react")),
    ]
    print_table(comp_rows, "Composite-9 (qwen)")

    # wydot bge_m3
    wb_base = RESULTS / "wydot_qwen_bge_m3_baselines.judged.jsonl"
    wydot_rows = [
        ("Monolithic",        stats(wb_base, filter_system="monolithic")),
        ("Regex-scoped",      stats(wb_base, filter_system="regex_scoped")),
        ("Hybrid-Routed",     stats(wb_base, filter_system="hybrid_routed")),
        ("MASDR-RAG",         stats(wb_base, filter_system="masdr_rag")),
        ("MASDR-SingleCall",  stats(RESULTS / "wydot_qwen_bge_m3_masdr_singlecall.judged.jsonl")),
    ]
    print_table(wydot_rows, "WYDOT (qwen, bge_m3)")

    # crag (placeholder until 53340743 lands)
    crag_rows = [
        ("MASDR-RAG",         stats(RESULTS / "crag_qwen_masdr_rag.judged.jsonl")),
    ]
    print_table(crag_rows, "CRAG/HotpotQA (qwen)")

    print("\n== LaTeX-ready: tab:singlecall body ==\n")
    print(latex_singlecall(comp_rows, "Composite-9"))
    print()
    print(latex_singlecall(wydot_rows, "WYDOT 200-query"))


if __name__ == "__main__":
    main()
