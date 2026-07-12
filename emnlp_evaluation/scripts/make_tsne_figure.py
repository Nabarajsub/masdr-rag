"""Generate the embedding-space dilution figure for the EMNLP paper.

Produces a 2-panel t-SNE projection of BGE-M3 chunk embeddings from the
composite enterprise corpus:

  (a) Monolithic retrieval space: all chunks, colored by source category.
      Visualizes cross-category overlap (the "dilution").
  (b) Scoped retrieval space: same 2-D coordinates, but only one agent's
      domain is highlighted; out-of-scope chunks are drawn faintly.

Output: PDF + PNG into acl-style-files-master/figures/.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DATA = Path("<DATA_ROOT>")
OUT_DIR = Path("<DATA_ROOT>")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 0
PER_CAT_CAP = 1200
PCA_DIM = 50
PERPLEXITY = 35
N_ITER = 1500

# Stable color map per source.
PALETTE = {
    "helpdesk":      "#d62728",  # red
    "confluence":    "#1f77b4",  # blue
    "stackoverflow": "#2ca02c",  # green
    "gmail":         "#9467bd",  # purple
    "docs":          "#ff7f0e",  # orange
}
PRETTY = {
    "helpdesk":      "Helpdesk",
    "confluence":    "Confluence",
    "stackoverflow": "StackOverflow",
    "gmail":         "Email (Enron)",
    "docs":          "Tech Docs",
}


def main() -> None:
    emb_path = DATA / "composite.embeddings.npy"
    meta_path = DATA / "composite.meta.parquet"
    print(f"[load] {emb_path}")
    emb = np.load(emb_path)
    meta = pd.read_parquet(meta_path)
    assert len(emb) == len(meta), (emb.shape, meta.shape)
    print(f"[load] embeddings={emb.shape}, meta={meta.shape}")

    rng = np.random.default_rng(SEED)
    idxs = []
    for cat, sub in meta.groupby("source_type", sort=False):
        keep = sub.index.to_numpy()
        if len(keep) > PER_CAT_CAP:
            keep = rng.choice(keep, size=PER_CAT_CAP, replace=False)
        idxs.append(keep)
    idxs = np.concatenate(idxs)
    rng.shuffle(idxs)
    sub_emb = emb[idxs].astype(np.float32)
    sub_meta = meta.iloc[idxs].reset_index(drop=True)
    print(f"[sample] {sub_emb.shape[0]} points")

    print(f"[pca] -> {PCA_DIM}d")
    pca = PCA(n_components=PCA_DIM, random_state=SEED)
    sub_emb_pca = pca.fit_transform(sub_emb)

    print(f"[t-sne] perplexity={PERPLEXITY} n_iter={N_ITER}")
    tsne = TSNE(
        n_components=2,
        perplexity=PERPLEXITY,
        max_iter=N_ITER,
        init="pca",
        random_state=SEED,
        learning_rate="auto",
        metric="cosine",
    )
    xy = tsne.fit_transform(sub_emb_pca)
    print(f"[t-sne] done: {xy.shape}")

    # Save intermediate so we can re-render without recomputing.
    np.savez(
        OUT_DIR / "tsne_coords.npz",
        xy=xy.astype(np.float32),
        source=sub_meta["source_type"].to_numpy(),
    )

    cats_ordered = ["helpdesk", "confluence", "stackoverflow", "gmail", "docs"]
    cats_present = [c for c in cats_ordered if c in set(sub_meta["source_type"])]
    src = sub_meta["source_type"].to_numpy()

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
    })

    # Panel (b): pick the agent that visually carves out a coherent region.
    spotlight = "stackoverflow"

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=True)

    # ---------- Panel (a): full mix ----------
    ax = axes[0]
    for cat in cats_present:
        mask = src == cat
        ax.scatter(
            xy[mask, 0], xy[mask, 1],
            s=4, alpha=0.55, linewidths=0,
            color=PALETTE[cat], label=PRETTY[cat], rasterized=True,
        )
    ax.set_title("(a) Monolithic search space", fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_edgecolor("#444")
    leg = ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, -0.18),
        ncol=3, frameon=False, handletextpad=0.3, columnspacing=0.8,
        markerscale=2.0,
    )

    # ---------- Panel (b): scoped to one agent ----------
    ax = axes[1]
    out_mask = src != spotlight
    in_mask = src == spotlight
    ax.scatter(
        xy[out_mask, 0], xy[out_mask, 1],
        s=4, alpha=0.10, linewidths=0,
        color="#999999", rasterized=True,
    )
    ax.scatter(
        xy[in_mask, 0], xy[in_mask, 1],
        s=6, alpha=0.85, linewidths=0,
        color=PALETTE[spotlight], rasterized=True,
    )
    ax.set_title(f"(b) Scoped: {PRETTY[spotlight]} agent only", fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_edgecolor("#444")
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=PALETTE[spotlight],
               markersize=6, label=f"in scope ({PRETTY[spotlight]})"),
        Line2D([0], [0], marker="o", linestyle="", color="#999999",
               alpha=0.35, markersize=6, label="out of scope (filtered)"),
    ]
    ax.legend(
        handles=handles,
        loc="lower center", bbox_to_anchor=(0.5, -0.18),
        ncol=2, frameon=False, handletextpad=0.3, columnspacing=0.8,
    )

    for ext in ("pdf", "png"):
        out = OUT_DIR / f"embedding_space_dilution.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"[save] {out}")
    plt.close(fig)

    counts = sub_meta["source_type"].value_counts().to_dict()
    print(f"[meta] points-per-source after sampling: {counts}")


if __name__ == "__main__":
    main()
