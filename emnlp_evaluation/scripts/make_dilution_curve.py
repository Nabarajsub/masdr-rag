"""Dilution-vs-scale figure: dilution factor delta against log(chunk count)
for the eight scopable WYDOT categories. Matches Table 1 in the paper."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

OUT = Path("<DATA_ROOT>")
OUT.mkdir(parents=True, exist_ok=True)

# (category, chunks, delta_mean) — from Table 1 of the paper (WYDOT).
ROWS = [
    ("Design Manual",        1_405,  0.53),
    ("Standard Specs",       2_519,  0.43),
    ("Materials Testing",    2_180,  0.22),
    ("Bridge Program",       5_399,  0.21),
    ("STIP",                13_634,  0.17),
    ("Traffic & Crashes",   30_922,  0.16),
    ("Annual Reports",       2_341,  0.12),
    ("Construction Manual",  6_641,  0.10),
]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
})

names = [r[0] for r in ROWS]
chunks = np.array([r[1] for r in ROWS], dtype=float)
delta  = np.array([r[2] for r in ROWS], dtype=float)
logc   = np.log10(chunks)

rho, p = spearmanr(chunks, delta)
m, b = np.polyfit(logc, delta, 1)
xs = np.linspace(logc.min() - 0.05, logc.max() + 0.05, 100)
ys = m * xs + b

fig, ax = plt.subplots(figsize=(3.4, 2.6), constrained_layout=True)
ax.plot(10 ** xs, ys, color="#888", linewidth=1.2, linestyle="--",
        label=f"linear fit (log-x)")
ax.scatter(chunks, delta, s=42, c="#1f77b4",
           edgecolors="#0a3a66", linewidths=0.6, zorder=3)

label_offsets = {
    "Design Manual":        (8, -13),
    "Standard Specs":       (8,  -2),
    "Materials Testing":    (8,   2),
    "Bridge Program":       (8,  -2),
    "STIP":                 (-95, 4),
    "Traffic & Crashes":    (-115,2),
    "Annual Reports":       (8,  -2),
    "Construction Manual":  (8,   2),
}
for name, c, d in zip(names, chunks, delta):
    dx, dy = label_offsets.get(name, (5, 5))
    ax.annotate(name, (c, d), xytext=(dx, dy), textcoords="offset points",
                fontsize=7, color="#222")

ax.set_xscale("log")
ax.set_xlabel("Chunks per category (log scale)")
ax.set_ylabel(r"Dilution factor $\delta$")
ax.set_xlim(1_000, 50_000)
ax.set_ylim(0.0, 0.62)
ax.grid(True, which="both", linestyle=":", linewidth=0.5, color="#ccc", alpha=0.7)
ax.set_axisbelow(True)
ax.text(0.97, 0.95,
        rf"Spearman $\rho={rho:.2f}$, $p={p:.2f}$" "\n" rf"$n={len(ROWS)}$",
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbb", lw=0.5))
ax.legend(loc="lower left", frameon=False)

for ext in ("pdf", "png"):
    out = OUT / f"dilution_vs_scale.{ext}"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[save] {out}")
plt.close(fig)
print(f"rho={rho:.3f} p={p:.3f}  slope={m:.3f}  intercept={b:.3f}")
