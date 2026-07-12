"""Re-render the WYDOT t-SNE figure from cached coords (no recompute).

Uses tsne_coords_wydot.npz produced by make_tsne_wydot_figure.py.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT_DIR = Path("<DATA_ROOT>")
COORDS = OUT_DIR / "tsne_coords_wydot.npz"

WYDOT_CATS = [
    "Standard Specs", "Construction Manual", "Materials Testing",
    "Design Manual", "Traffic & Crashes", "Bridge Program",
    "STIP", "Annual Reports", "Highway Safety",
]
PALETTE = {
    "Standard Specs":      "#1f77b4",
    "Construction Manual": "#ff7f0e",
    "Materials Testing":   "#2ca02c",
    "Design Manual":       "#d62728",
    "Traffic & Crashes":   "#9467bd",
    "Bridge Program":      "#8c564b",
    "STIP":                "#e377c2",
    "Annual Reports":      "#7f7f7f",
    "Highway Safety":      "#bcbd22",
}


def main() -> None:
    d = np.load(COORDS, allow_pickle=True)
    xy = d["xy"]
    cats = d["category"]
    print(f"[load] {xy.shape[0]} points, {len(set(cats))} categories")
    print(f"        present: {sorted(set(cats))}")

    plt.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "legend.fontsize": 6.5, "pdf.fonttype": 42,
    })
    cats_present = [c for c in WYDOT_CATS if c in set(cats)]
    spotlight = "Standard Specs"

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), constrained_layout=True)

    ax = axes[0]
    for cat in cats_present:
        m = cats == cat
        ax.scatter(xy[m, 0], xy[m, 1], s=4, alpha=0.55, linewidths=0,
                   color=PALETTE.get(cat, "#666"), label=cat, rasterized=True)
    ax.set_title("(a) WYDOT monolithic search space", fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_linewidth(0.6); sp.set_edgecolor("#444")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.40),
              ncol=3, frameon=False, handletextpad=0.3, columnspacing=0.7,
              markerscale=2.0)

    ax = axes[1]
    out_mask = cats != spotlight
    in_mask = cats == spotlight
    ax.scatter(xy[out_mask, 0], xy[out_mask, 1],
               s=4, alpha=0.10, linewidths=0, color="#999", rasterized=True)
    ax.scatter(xy[in_mask, 0], xy[in_mask, 1],
               s=6, alpha=0.85, linewidths=0, color=PALETTE[spotlight],
               rasterized=True)
    ax.set_title(f"(b) WYDOT scoped: {spotlight} agent only", fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_linewidth(0.6); sp.set_edgecolor("#444")
    ax.legend(handles=[
        Line2D([0], [0], marker="o", linestyle="", color=PALETTE[spotlight],
               markersize=6, label=f"in scope ({spotlight})"),
        Line2D([0], [0], marker="o", linestyle="", color="#999", alpha=0.35,
               markersize=6, label="out of scope (filtered)"),
    ], loc="lower center", bbox_to_anchor=(0.5, -0.22),
              ncol=2, frameon=False, handletextpad=0.3, columnspacing=0.8)

    for ext in ("pdf", "png"):
        out = OUT_DIR / f"embedding_space_dilution_wydot.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"[save] {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
