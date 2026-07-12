"""Retrieval-source confusion heatmap.

For each system (monolithic vs regex_scoped vs hybrid_routed) compute the
distribution P(retrieved_chunk_source | gold_source) over all queries in
the composite benchmark. Visualizes how monolithic retrieval mixes
sources, and how scoping collapses retrieval onto the diagonal.
"""
from __future__ import annotations
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

INFILE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "<DATA_ROOT>"
    "emnlp_evaluation/results/"
    "composite_qwen_monolithic-regex_scoped-hybrid_routed-masdr_rag-react"
    ".judged.judged.judged.judged.jsonl"
)
OUT_SUFFIX = sys.argv[2] if len(sys.argv) > 2 else ""
OUT = Path(
    "<DATA_ROOT>"
    "acl-style-files-master/figures"
)
OUT.mkdir(parents=True, exist_ok=True)

SOURCES = ["helpdesk", "confluence", "stackoverflow", "gmail", "docs"]
PRETTY = {
    "helpdesk":      "Helpdesk",
    "confluence":    "Confluence",
    "stackoverflow": "StackOverflow",
    "gmail":         "Email",
    "docs":          "Docs",
}
SYSTEMS_TO_PLOT = [
    ("monolithic",    "(a) Monolithic"),
    ("regex_scoped",  "(b) Regex-Scoped"),
    ("hybrid_routed", "(c) Hybrid-Routed"),
]


def build_matrix(records: list[dict], system: str) -> tuple[np.ndarray, dict]:
    by_gold = defaultdict(Counter)
    n_by_gold = Counter()
    for r in records:
        if r.get("system") != system:
            continue
        gold = r.get("category")
        if gold not in SOURCES:
            continue
        retr = r.get("chunk_sources") or []
        if not retr:
            continue
        n_by_gold[gold] += 1
        for s in retr:
            if s in SOURCES:
                by_gold[gold][s] += 1
    M = np.zeros((len(SOURCES), len(SOURCES)), dtype=float)
    for i, gi in enumerate(SOURCES):
        total = sum(by_gold[gi].values())
        if total == 0:
            continue
        for j, gj in enumerate(SOURCES):
            M[i, j] = by_gold[gi][gj] / total
    return M, n_by_gold


def main() -> None:
    print(f"[load] {INFILE}")
    records = [json.loads(l) for l in INFILE.open()]
    print(f"[load] {len(records)} records")

    cmap = LinearSegmentedColormap.from_list(
        "blues",
        ["#ffffff", "#cfe1f2", "#6baed6", "#2171b5", "#08306b"],
    )

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), constrained_layout=True)
    n_labels = [PRETTY[s] for s in SOURCES]

    for ax, (sys_name, title) in zip(axes, SYSTEMS_TO_PLOT):
        M, ngold = build_matrix(records, sys_name)
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="equal")
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(range(len(SOURCES)), n_labels, rotation=45,
                      ha="right", fontsize=7)
        ax.set_yticks(range(len(SOURCES)), n_labels, fontsize=7)
        if ax is axes[0]:
            ax.set_ylabel("Query gold source", fontsize=8)
        ax.set_xlabel("Retrieved chunk source", fontsize=8)
        for i in range(len(SOURCES)):
            for j in range(len(SOURCES)):
                val = M[i, j]
                if val < 0.005:
                    continue
                color = "white" if val > 0.55 else "#111"
                ax.text(j, i, f"{val:.2f}".lstrip("0") if val < 1 else "1.0",
                        ha="center", va="center", fontsize=6.5, color=color)
        diag = float(np.trace(M)) / len(SOURCES)
        ax.text(0.5, -0.42, f"diag avg = {diag:.2f}",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=7, color="#444")

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("P(retrieved source $\\mid$ gold source)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    for ext in ("pdf", "png"):
        out = OUT / f"retrieval_confusion{OUT_SUFFIX}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"[save] {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
