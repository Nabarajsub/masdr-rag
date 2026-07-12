"""Embedding-space dilution figure for WYDOT.

Mirrors make_tsne_figure.py but uses the WYDOT corpus (9 source labels).
Outputs:
    figures/embedding_space_dilution_wydot.pdf  (2 panels: monolithic + STANDARD_SPECS scope)
    figures/tsne_coords_wydot.npz
"""
from __future__ import annotations

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
PER_CAT_CAP = 800           # 9 cats × 800 = 7200 max — keeps t-SNE legible
PCA_DIM = 50
PERPLEXITY = 35
N_ITER = 1500

# 9 WYDOT source types. Color palette balanced for distinguishability.
WYDOT_CATS = [
    "Standard Specs",
    "Construction Manual",
    "Materials Testing",
    "Design Manual",
    "Traffic & Crashes",
    "Bridge Program",
    "STIP",
    "Annual Reports",
    "Highway Safety",
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

# Map raw source_type field (which holds doc title fragments) to a category label.
# WYDOT meta uses 'source_type' field where the WYDOT FAISS dump puts the
# document.source — typically a path or short name. We map heuristically.
KEYWORDS = [
    ("Standard Specs",      ["standard.spec", "specifications"]),
    ("Construction Manual", ["construction.manual", "construction"]),
    ("Materials Testing",   ["material", "mtm"]),
    ("Design Manual",       ["design.manual", "cadd", "alignment"]),
    ("Traffic & Crashes",   ["crash", "traffic"]),
    ("Bridge Program",      ["bridge"]),
    ("STIP",                ["stip", "corridor"]),
    ("Annual Reports",      ["annual", "financial"]),
    ("Highway Safety",      ["safety", "shsp"]),
]


def _to_cat(src: str) -> str | None:
    s = (src or "").lower()
    for cat, kws in KEYWORDS:
        for kw in kws:
            if kw in s:
                return cat
    return None


def main() -> None:
    emb_path = DATA / "wydot.embeddings.npy"
    meta_path = DATA / "wydot.meta.parquet"
    print(f"[load] {emb_path}")
    emb = np.load(emb_path)
    meta = pd.read_parquet(meta_path)
    print(f"[load] embeddings={emb.shape}, meta={meta.shape}, cols={list(meta.columns)[:8]}")

    # Use whichever WYDOT field carries the doc identifier.
    src_field = "source_type" if "source_type" in meta.columns else "title"
    meta["category"] = meta[src_field].astype(str).apply(_to_cat)
    n_unmapped = int(meta["category"].isna().sum())
    print(f"[map] {len(meta) - n_unmapped}/{len(meta)} chunks mapped to a category")
    meta_known = meta.dropna(subset=["category"]).reset_index().rename(columns={"index": "orig_idx"})
    keep_idx = meta_known["orig_idx"].to_numpy()
    print(f"[map] keeping {len(keep_idx)} chunks across {meta_known['category'].nunique()} categories")
    print(f"      per-cat counts: {dict(meta_known['category'].value_counts())}")

    rng = np.random.default_rng(SEED)
    sample_idxs = []
    for cat, sub in meta_known.groupby("category", sort=False):
        keep = sub.index.to_numpy()
        if len(keep) > PER_CAT_CAP:
            keep = rng.choice(keep, size=PER_CAT_CAP, replace=False)
        sample_idxs.append(keep)
    sample_idxs = np.concatenate(sample_idxs)
    rng.shuffle(sample_idxs)

    orig_positions = meta_known.iloc[sample_idxs]["orig_idx"].to_numpy()
    sub_emb = emb[orig_positions].astype(np.float32)
    sub_meta = meta_known.iloc[sample_idxs].reset_index(drop=True)
    print(f"[sample] {sub_emb.shape[0]} points")

    print(f"[pca] -> {PCA_DIM}d")
    sub_emb_pca = PCA(n_components=PCA_DIM, random_state=SEED).fit_transform(sub_emb)

    print(f"[t-sne] perplexity={PERPLEXITY}")
    xy = TSNE(
        n_components=2, perplexity=PERPLEXITY, max_iter=N_ITER,
        init="pca", random_state=SEED, learning_rate="auto", metric="cosine",
    ).fit_transform(sub_emb_pca)
    print(f"[t-sne] done: {xy.shape}")

    np.savez(
        OUT_DIR / "tsne_coords_wydot.npz",
        xy=xy.astype(np.float32),
        category=sub_meta["category"].to_numpy(),
    )

    plt.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "legend.fontsize": 6.5, "pdf.fonttype": 42,
    })
    src = sub_meta["category"].to_numpy()
    cats_present = [c for c in WYDOT_CATS if c in set(src)]

    spotlight = "Standard Specs"
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=True)

    # ---------- Panel (a): full mix ----------
    ax = axes[0]
    for cat in cats_present:
        m = src == cat
        ax.scatter(xy[m, 0], xy[m, 1], s=4, alpha=0.55, linewidths=0,
                   color=PALETTE.get(cat, "#666666"), label=cat, rasterized=True)
    ax.set_title("(a) WYDOT monolithic search space", fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_linewidth(0.6); sp.set_edgecolor("#444")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32),
              ncol=3, frameon=False, handletextpad=0.3, columnspacing=0.7,
              markerscale=2.0)

    # ---------- Panel (b): one agent's scope active ----------
    ax = axes[1]
    out_mask = src != spotlight
    in_mask = src == spotlight
    ax.scatter(xy[out_mask, 0], xy[out_mask, 1],
               s=4, alpha=0.10, linewidths=0, color="#999999", rasterized=True)
    ax.scatter(xy[in_mask, 0], xy[in_mask, 1],
               s=6, alpha=0.85, linewidths=0,
               color=PALETTE.get(spotlight, "#1f77b4"), rasterized=True)
    ax.set_title(f"(b) WYDOT scoped: {spotlight} agent only", fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_linewidth(0.6); sp.set_edgecolor("#444")
    ax.legend(handles=[
        Line2D([0], [0], marker="o", linestyle="", color=PALETTE[spotlight],
               markersize=6, label=f"in scope ({spotlight})"),
        Line2D([0], [0], marker="o", linestyle="", color="#999999", alpha=0.35,
               markersize=6, label="out of scope (filtered)"),
    ], loc="lower center", bbox_to_anchor=(0.5, -0.18),
              ncol=2, frameon=False, handletextpad=0.3, columnspacing=0.8)

    for ext in ("pdf", "png"):
        out = OUT_DIR / f"embedding_space_dilution_wydot.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"[save] {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
