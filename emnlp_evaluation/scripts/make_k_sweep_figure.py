"""Build a R@k-vs-k line plot per (corpus, system) from judge_assets/k_sweep.json."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt

DATA = Path("<DATA_ROOT>")
OUT = Path("<DATA_ROOT>")
OUT.mkdir(parents=True, exist_ok=True)

CORPORA = ["wydot_bge", "composite", "multihop", "wydot_gem"]
SYS_KEEP = ["monolithic", "regex_scoped", "hybrid_routed", "masdr_rag"]
SYS_COLOR = {
    "monolithic": "#888",
    "regex_scoped": "#4c9be8",
    "hybrid_routed": "#2ca02c",
    "masdr_rag": "#9467bd",
}
SYS_PRETTY = {
    "monolithic": "Monolithic",
    "regex_scoped": "Regex-Scoped",
    "hybrid_routed": "Hybrid-Routed",
    "masdr_rag": "MASDR-RAG",
}
CORPUS_PRETTY = {
    "wydot_bge": "WYDOT (BGE-M3)",
    "composite": "Composite-9",
    "multihop": "MultiHop-RAG",
    "wydot_gem": "WYDOT (Gemini)",
}


def main():
    blob = json.loads(DATA.read_text())
    plt.rcParams.update({"font.family": "serif", "font.size": 9,
                         "axes.titlesize": 9.5, "legend.fontsize": 7,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, len(CORPORA), figsize=(8.2, 2.6),
                             sharey=True, constrained_layout=True)
    ks = [3, 5, 10, 15]
    for ax, corpus in zip(axes, CORPORA):
        for sys_key in SYS_KEEP:
            key = f"{corpus}/{sys_key}"
            row = blob.get(key)
            if not row:
                continue
            ys = [row.get(f"R@{k}") for k in ks]
            if any(y is None for y in ys):
                continue
            ax.plot(ks, ys, marker="o", color=SYS_COLOR[sys_key],
                    label=SYS_PRETTY[sys_key], linewidth=1.2, markersize=4)
        ax.set_title(CORPUS_PRETTY[corpus])
        ax.set_xticks(ks)
        ax.set_xlabel("$k$")
        ax.grid(True, linestyle=":", linewidth=0.5, color="#ccc")
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("Recall@$k$")
    axes[-1].legend(loc="lower right", frameon=False, fontsize=6.5)
    for ext in ("pdf", "png"):
        out = OUT / f"k_sweep.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"[save] {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
