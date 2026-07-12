"""Publication-quality Pareto plot: latency p50 vs correctness for all
systems and retrieval-only baselines on EnterpriseComposite-9, Qwen-2.5-7B
synthesiser, BGE-M3 retriever (where applicable), Qwen-judge."""
from __future__ import annotations
import json, glob
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path("<DATA_ROOT>")
OUT = Path("<DATA_ROOT>")
OUT.mkdir(parents=True, exist_ok=True)

# (label_for_plot, glob_pattern, internal_system_name, category)
ENTRIES = [
    ("BM25-only",        f"{ROOT}/composite_bm25_*.judged*.jsonl",     "bm25_only",     "retriever"),
    ("ColBERTv2-only",   f"{ROOT}/composite_colbert_*.judged*.jsonl",  "colbert_only",  "retriever"),
    ("BM25 + Qwen",      f"{ROOT}/composite_bm25_*.judged*.jsonl",     "bm25_qwen",     "retr_llm"),
    ("ColBERTv2 + Qwen", f"{ROOT}/composite_colbert_*.judged*.jsonl",  "colbert_qwen",  "retr_llm"),
    ("LangChain ReAct",  f"{ROOT}/composite_langchain_*.judged*.jsonl","langchain",     "agentic"),
    ("Monolithic",       f"{ROOT}/composite_qwen_*.judged*.jsonl",     "monolithic",    "scoped"),
    ("Regex-Scoped",     f"{ROOT}/composite_qwen_*.judged*.jsonl",     "regex_scoped",  "scoped"),
    ("Hybrid-Routed",    f"{ROOT}/composite_qwen_*.judged*.jsonl",     "hybrid_routed", "scoped"),
    ("Custom ReAct",     f"{ROOT}/composite_qwen_*.judged*.jsonl",     "react",         "agentic"),
]

COLORS = {"retriever":"#9b9b9b", "retr_llm":"#1f77b4", "scoped":"#2ca02c", "agentic":"#d62728"}
MARKERS = {"retriever":"s", "retr_llm":"o", "scoped":"D", "agentic":"^"}


def latest_judged(pattern: str) -> str:
    files = glob.glob(pattern)
    if not files: return ""
    # most-judged version (most ".judged." in name) wins.
    return max(files, key=lambda p: (p.count(".judged."), Path(p).stat().st_mtime))


def aggregate(label: str, pattern: str, sysname: str, cat: str):
    f = latest_judged(pattern)
    if not f: return None
    corr, lat = [], []
    with open(f) as fh:
        for line in fh:
            try: r = json.loads(line)
            except: continue
            if r.get("system") != sysname: continue
            c = r.get("correctness")
            if isinstance(c, (int, float)): corr.append(float(c))
            w = r.get("wall_time_s")
            if isinstance(w, (int, float)): lat.append(float(w))
    if not corr or not lat: return None
    return {
        "label": label, "category": cat,
        "n": len(corr),
        "corr": float(np.mean(corr)),
        "lat_p50": float(np.median(lat)),
        "lat_p25": float(np.percentile(lat, 25)),
        "lat_p75": float(np.percentile(lat, 75)),
    }


def pareto_frontier(points):
    pts = sorted(points, key=lambda p: (p["lat_p50"], -p["corr"]))
    frontier, best = [], -1
    for p in pts:
        if p["corr"] > best:
            frontier.append(p); best = p["corr"]
    return frontier


def main() -> None:
    rows = [aggregate(*e) for e in ENTRIES]
    rows = [r for r in rows if r is not None]
    for r in rows:
        print(f"  {r['label']:>18}  n={r['n']:>3}  corr={r['corr']:.3f}  lat50={r['lat_p50']:5.2f}s")

    plt.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "legend.fontsize": 7, "pdf.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(7.0, 3.2), constrained_layout=True)

    # Pareto frontier
    frontier = pareto_frontier(rows)
    xs = [p["lat_p50"] for p in frontier]
    ys = [p["corr"] for p in frontier]
    ax.plot(xs, ys, color="#bbbbbb", linewidth=1.2, linestyle="--",
            zorder=1, label="Pareto frontier")

    # Points
    for r in rows:
        ax.errorbar(
            r["lat_p50"], r["corr"],
            xerr=[[r["lat_p50"] - r["lat_p25"]], [r["lat_p75"] - r["lat_p50"]]],
            fmt=MARKERS[r["category"]], color=COLORS[r["category"]],
            markersize=7, markeredgecolor="black", markeredgewidth=0.5,
            ecolor=COLORS[r["category"]], elinewidth=0.7, capsize=2, alpha=0.95,
            zorder=3,
        )

    # Custom label placement to avoid collisions on the log-x axis.
    offsets = {
        "BM25-only":        (-32,  -16),
        "ColBERTv2-only":   ( 6,    -2),
        "BM25 + Qwen":      ( 8,    -4),
        "ColBERTv2 + Qwen": ( 8,     6),
        "LangChain ReAct":  (-15,  -16),
        "Monolithic":       (-50,    8),
        "Regex-Scoped":     ( 8,    -3),
        "Hybrid-Routed":    ( 8,    -8),
        "Custom ReAct":     ( 8,     4),
    }
    for r in rows:
        dx, dy = offsets.get(r["label"], (6, 6))
        ax.annotate(r["label"], (r["lat_p50"], r["corr"]),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=7.5, color="#222")

    ax.set_xlabel("Latency p50 (seconds)")
    ax.set_ylabel("Correctness")
    ax.set_xscale("log")
    ax.set_xlim(0.005, 30)
    ax.set_ylim(0.35, 1.02)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, color="#ccc", alpha=0.6)
    ax.set_axisbelow(True)

    legend_handles = [
        Line2D([0], [0], marker=MARKERS["retriever"], color="w",
               markerfacecolor=COLORS["retriever"], markersize=7,
               markeredgecolor="black", markeredgewidth=0.5,
               label="Retriever only"),
        Line2D([0], [0], marker=MARKERS["retr_llm"], color="w",
               markerfacecolor=COLORS["retr_llm"], markersize=7,
               markeredgecolor="black", markeredgewidth=0.5,
               label="Retriever $+$ Qwen"),
        Line2D([0], [0], marker=MARKERS["scoped"], color="w",
               markerfacecolor=COLORS["scoped"], markersize=7,
               markeredgecolor="black", markeredgewidth=0.5,
               label="Scoped / routed"),
        Line2D([0], [0], marker=MARKERS["agentic"], color="w",
               markerfacecolor=COLORS["agentic"], markersize=7,
               markeredgecolor="black", markeredgewidth=0.5,
               label="Iterative agent"),
        Line2D([0], [0], color="#bbbbbb", linestyle="--", linewidth=1.2,
               label="Pareto frontier"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True,
              fancybox=False, edgecolor="#bbb", handletextpad=0.4,
              ncol=2, columnspacing=0.8, fontsize=7.5)

    for ext in ("pdf", "png"):
        out = OUT / f"pareto.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"[save] {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
