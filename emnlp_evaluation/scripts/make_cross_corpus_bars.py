"""Grouped bar chart of per-system correctness across three benchmarks:
WYDOT (production proprietary, Qwen-stack judged), EnterpriseComposite-9
(public, Qwen-judged), and CRAG-style HotpotQA-distractor (Qwen-judged)."""
from __future__ import annotations
import json, glob
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("<DATA_ROOT>")
OUT = Path("<DATA_ROOT>")
OUT.mkdir(parents=True, exist_ok=True)

# (pretty_name, internal_system)
SYSTEMS = [
    ("Monolithic",     "monolithic"),
    ("Regex-Scoped",   "regex_scoped"),
    ("Hybrid-Routed",  "hybrid_routed"),
    ("MASDR-RAG",      "masdr_rag"),
    ("Custom ReAct",   "react"),
]
SYS_COLORS = {
    "Monolithic":     "#888888",
    "Regex-Scoped":   "#4c9be8",
    "Hybrid-Routed":  "#2ca02c",
    "MASDR-RAG":      "#9467bd",
    "Custom ReAct":   "#d62728",
}

# v2: scoped systems come from the fixed-router re-runs. Because corr_for()
# dedupes by query_id with later files overwriting earlier ones, the corrected
# fixedrouter files are listed LAST so they take precedence; ReAct rows
# (unaffected by the router bug) survive from the original runs.
BENCHMARKS = [
    ("WYDOT",            [
        f"{ROOT}/wydot_qwen_bge_m3_baselines.judged.jsonl",
        f"{ROOT}/wydot_qwen_gemini_*_shard*.judged.jsonl",
    ]),
    ("Composite-9",      [
        f"{ROOT}/composite_qwen_monolithic-regex_scoped-hybrid_routed-masdr_rag-react.judged.jsonl",
        f"{ROOT}/composite_qwen_fixedrouter_monolithic-regex_scoped-hybrid_routed-masdr_rag.judged.jsonl",
    ]),
    ("MultiHop-RAG",     [
        f"{ROOT}/multihop_qwen_react.judged.jsonl",
        f"{ROOT}/multihop_qwen_fixedrouter_monolithic-regex_scoped-hybrid_routed-masdr_rag.judged.jsonl",
    ]),
    ("HotpotQA (CRAG)",  [
        f"{ROOT}/crag_qwen_monolithic-hybrid_routed-masdr_rag-react.judged.jsonl",
        f"{ROOT}/crag_qwen_fixedrouter_monolithic-regex_scoped.judged.jsonl",
        f"{ROOT}/crag_qwen_fixedrouter_hybrid_routed-masdr_rag.judged.jsonl",
    ]),
]


def latest_files(patterns):
    files = []
    for p in patterns:
        files += glob.glob(p)
    return files


def corr_for(patterns, sysname):
    """Mean correctness for one (benchmark, system) pair, deduped by query_id
    so multi-judged files don't double-count."""
    by_qid: dict[str, float] = {}
    for f in latest_files(patterns):
        with open(f) as fh:
            for line in fh:
                try: r = json.loads(line)
                except: continue
                if r.get("system") != sysname: continue
                c = r.get("correctness")
                if not isinstance(c, (int, float)): continue
                qid = r.get("query_id") or f"_{len(by_qid)}"
                by_qid[qid] = float(c)
    if not by_qid: return None, 0
    return float(np.mean(list(by_qid.values()))), len(by_qid)


def main():
    data = {}  # data[bench][sys] = (mean, n)
    for bench, pats in BENCHMARKS:
        data[bench] = {}
        for pretty, sysname in SYSTEMS:
            m, n = corr_for(pats, sysname)
            data[bench][pretty] = (m, n)
            print(f"  {bench:>18}  {pretty:>15}  n={n:>4}  corr={('n/a' if m is None else f'{m:.3f}')}")

    plt.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "legend.fontsize": 7.5, "pdf.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(7.5, 3.0), constrained_layout=True)
    bench_names = [b[0] for b in BENCHMARKS]
    n_bench = len(bench_names)
    n_sys = len(SYSTEMS)
    bw = 0.15
    x = np.arange(n_bench)

    for i, (pretty, _) in enumerate(SYSTEMS):
        means = []
        for b in bench_names:
            m, _ = data[b][pretty]
            means.append(m if m is not None else 0.0)
        xs = x + (i - (n_sys - 1) / 2) * bw
        bars = ax.bar(xs, means, width=bw, color=SYS_COLORS[pretty],
                      edgecolor="black", linewidth=0.4, label=pretty)
        for xi, mi, b in zip(xs, means, bench_names):
            m, n = data[b][pretty]
            if m is None:
                ax.text(xi, 0.01, "n/a", ha="center", va="bottom",
                        fontsize=6.5, color="#666", rotation=90)
            else:
                ax.text(xi, mi + 0.008, f"{mi*100:.0f}", ha="center",
                        va="bottom", fontsize=6.5, color="#222")

    ax.set_xticks(x, bench_names)
    ax.set_ylabel("Correctness")
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, color="#ccc", alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=5,
              frameon=False, handlelength=1.2, handletextpad=0.4,
              columnspacing=0.8)

    for ext in ("pdf", "png"):
        out = OUT / f"cross_corpus_bars.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"[save] {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
