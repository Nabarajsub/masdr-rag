"""Aggregate rerank and Neo4j-vs-FAISS comparisons.

When the rerank jobs (53346191--96) and Neo4j-backed jobs (53346186--90)
finish, this produces the LaTeX-ready rows for:
  Table tab:rerank        — Δ(rerank − base) per (corpus, system)
  Table tab:neo4j_vs_faiss — Faith/Corr/R@10 per (corpus, system, backend)

Reads paired .judged.jsonl files; if a paired file is missing, the row
is emitted as "---" placeholders.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

RESULTS = Path("<DATA_ROOT>")

CORPORA = ["wydot", "composite", "multihop", "financebench", "mmlu_pro", "nq"]
SYSTEMS = ["monolithic", "regex_scoped", "hybrid_routed", "masdr_rag"]


def _gold_hit(r: dict, k: int = 10) -> int | None:
    g = r.get("gold_chunk_id")
    if g:
        return 1 if g in (r.get("chunk_ids") or [])[:k] else 0
    titles = r.get("gold_titles") or []
    if titles:
        gset = {t.strip().lower() for t in titles if t}
        rset = {(t or "").strip().lower() for t in (r.get("chunk_titles") or r.get("chunk_series") or [])[:k] if t}
        return 1 if (gset & rset) else 0
    rt = r.get("relevant_title")
    if rt:
        ser = [(t or "").strip().lower() for t in (r.get("chunk_series") or [])[:k]]
        return 1 if rt.strip().lower() in ser else 0
    return None


def stats(path: Path, system_filter: str | None = None):
    by = defaultdict(lambda: dict(n=0, f=[], c=[], h=[]))
    seen = set()
    if not path.exists():
        return None
    for line in path.open():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "error" in r:
            continue
        s = r.get("system", "?")
        if system_filter and s != system_filter:
            continue
        k = (r.get("query_id"), s)
        if k in seen:
            continue
        seen.add(k)
        d = by[s]
        d["n"] += 1
        if r.get("faithfulness") is not None:
            d["f"].append(float(r["faithfulness"]))
        if r.get("correctness") is not None:
            d["c"].append(float(r["correctness"]))
        h = _gold_hit(r)
        if h is not None:
            d["h"].append(h)
    avg = lambda L: sum(L) / max(1, len(L)) if L else None
    return {s: {"n": v["n"], "faith": avg(v["f"]), "corr": avg(v["c"]),
                "r10": avg(v["h"])} for s, v in by.items()}


def _find_judged(prefix: str) -> Path | None:
    matches = sorted(RESULTS.glob(f"{prefix}*.judged.jsonl"))
    matches = [m for m in matches if ".judged.judged" not in m.name]
    return matches[0] if matches else None


def rerank_table() -> str:
    """LaTeX rows for the rerank delta table."""
    rows = []
    for sys_label, sys_key in [("Monolithic", "monolithic"), ("Hybrid-R", "hybrid_routed")]:
        rows.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{sys_label}}}}} \\\\")
        for corpus in CORPORA:
            base_path = _find_judged(f"{corpus}_qwen_") if corpus != "wydot" \
                        else _find_judged("wydot_qwen_bge_m3_baselines")
            rerank_path = _find_judged(f"{corpus}_qwen_rerank") if corpus != "wydot" \
                          else _find_judged("wydot_qwen_bge_m3_rerank")
            base = stats(base_path, system_filter=sys_key) if base_path else None
            rer = stats(rerank_path, system_filter=sys_key) if rerank_path else None
            if not base or not rer or sys_key not in base or sys_key not in rer:
                rows.append(f"\\quad {corpus} & --- & --- & --- \\\\")
                continue
            b, r = base[sys_key], rer[sys_key]
            df = (r["faith"] or 0) - (b["faith"] or 0) if b["faith"] is not None and r["faith"] is not None else None
            dc = (r["corr"] or 0) - (b["corr"] or 0) if b["corr"] is not None and r["corr"] is not None else None
            dr = (r["r10"] or 0) - (b["r10"] or 0) if b["r10"] is not None and r["r10"] is not None else None
            def fmt(x): return "---" if x is None else f"${x:+.3f}$"
            rows.append(f"\\quad {corpus:<13} & {fmt(df)} & {fmt(dc)} & {fmt(dr)} \\\\")
    return "\n".join(rows)


def neo4j_vs_faiss_table() -> str:
    """LaTeX rows for the Neo4j-vs-FAISS comparison table."""
    rows = []
    for corpus in CORPORA:
        if corpus == "wydot":
            faiss_path = _find_judged("wydot_qwen_bge_m3_baselines")
            neo4j_path = None  # WYDOT uses production Neo4j; covered elsewhere
        else:
            faiss_path = _find_judged(f"{corpus}_qwen_monolithic")
            neo4j_path = _find_judged(f"{corpus}_qwen_neo4j_monolithic")
        for sys_key in SYSTEMS:
            f = stats(faiss_path, sys_key) if faiss_path else None
            n = stats(neo4j_path, sys_key) if neo4j_path else None
            f_s = f.get(sys_key) if f else None
            n_s = n.get(sys_key) if n else None
            def cell(d, key):
                if not d or d.get(key) is None: return "---"
                return f"${d[key]:.3f}$"
            rows.append(
                f"{corpus:<13} & {sys_key:<14} & "
                f"{cell(f_s,'faith')} & {cell(n_s,'faith')} & "
                f"{cell(f_s,'corr')} & {cell(n_s,'corr')} & "
                f"{cell(f_s,'r10')} & {cell(n_s,'r10')} \\\\"
            )
    return "\n".join(rows)


def main():
    print("=== Rerank Δ-table ===")
    print(rerank_table())
    print()
    print("=== Neo4j vs FAISS table ===")
    print(neo4j_vs_faiss_table())


if __name__ == "__main__":
    main()
