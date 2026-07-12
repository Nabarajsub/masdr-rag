"""S-1: per-query dilution analysis on the WYDOT 200-query suite.

The original n=8 category-aggregate Spearman (rho = -0.60, p = 0.12)
is not statistically significant. This script lifts the analysis to per
query and shows the effect with 10x the effective sample size.

Each WYDOT query has:
    relevant_title : str (gold document title, e.g. "2021 Standard
                     Specifications for Road and Bridge Construction")
    category       : agent label (STANDARD_SPECS, CONSTRUCTION_MANUAL,
                     ...)
    chunk_series   : list of doc-series strings for the top-k retrieved
                     chunks

For each query q, system s, and k in {3, 5, 10, 15}, a "hit" is:
    hit_s,k(q) = 1 if relevant_title in chunk_series[:k] else 0

The per-query dilution is then
    delta_q,k = hit_scoped,k(q) - hit_global,k(q)
and we run paired tests + per-query Pearson against log_density(category).

Inputs:
    results/wydot_qwen_gemini_*_shardNofN.judged.jsonl    (4 shards)

Outputs:
    figures/perquery_dilution.pdf  (paper figure)
    figures/perquery_dilution_stats.json
"""
from __future__ import annotations

import glob, json, re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, pearsonr, spearmanr

ROOT = Path("<DATA_ROOT>")
FIG_DIR = Path("<DATA_ROOT>")
FIG_DIR.mkdir(parents=True, exist_ok=True)

INFILES = sorted(glob.glob(str(ROOT / "wydot_qwen_gemini_*_shard*.judged.jsonl")))

# Per-agent chunk counts on WYDOT (from Table 3 in the paper).
AGENT_CHUNK_COUNTS = {
    "STANDARD_SPECS":      3140,
    "CONSTRUCTION_MANUAL": 6641,
    "MATERIALS_TESTING":   2184,
    "DESIGN_MANUAL":       1366,
    "TRAFFIC_CRASHES":    30922,
    "BRIDGE_PROGRAM":      8076,
    "STIP":               13607,
    "ANNUAL_REPORT":       2439,
    "HIGHWAY_SAFETY":     30922,
}


def hit_at_k(chunk_series: list[str], gold_title: str, k: int) -> int:
    if not gold_title or not chunk_series:
        return 0
    target = gold_title.strip().lower()
    head = [(s or "").strip().lower() for s in chunk_series[:k]]
    return 1 if target in head else 0


def main() -> None:
    print(f"[load] {len(INFILES)} shards")
    by_q = defaultdict(dict)
    for f in INFILES:
        for line in open(f):
            try: r = json.loads(line)
            except: continue
            qid = r.get("query_id")
            sys = r.get("system")
            if qid and sys: by_q[qid][sys] = r
    print(f"[load] {len(by_q)} unique queries")

    rows = []
    for qid, by_sys in by_q.items():
        mono = by_sys.get("monolithic")
        scop = by_sys.get("regex_scoped") or by_sys.get("hybrid_routed")
        if not mono or not scop: continue
        gold_title = mono.get("relevant_title")
        cat = mono.get("category")
        if cat not in AGENT_CHUNK_COUNTS or not gold_title: continue
        for k in (3, 5, 10, 15):
            p_g = hit_at_k(mono.get("chunk_series") or [], gold_title, k)
            p_s = hit_at_k(scop.get("chunk_series") or [], gold_title, k)
            rows.append({
                "qid": qid, "category": cat, "k": k,
                "p_global": p_g, "p_scoped": p_s,
                "delta": p_s - p_g,
                "log_density": float(np.log10(AGENT_CHUNK_COUNTS[cat])),
            })
    print(f"per-query observations: {len(rows)}")
    print(f"unique queries used: {len(set(r['qid'] for r in rows))}")

    # ---- 1. Wilcoxon paired test at each k ----
    print("\n=== Wilcoxon (paired) hit_scoped vs hit_global at each k ===")
    wilcox = {}
    for k in (3, 5, 10, 15):
        sub = [r for r in rows if r["k"] == k]
        diffs = [r["delta"] for r in sub]
        nz = [d for d in diffs if d != 0]
        if len(nz) < 5:
            print(f"  k={k}: too few non-zero diffs ({len(nz)})")
            wilcox[k] = {"n": len(sub), "n_nonzero": len(nz)}
            continue
        stat, p = wilcoxon(nz, alternative="two-sided", zero_method="pratt")
        wilcox[k] = {"n": int(len(sub)), "n_nonzero": int(len(nz)),
                     "mean_delta": float(np.mean(diffs)),
                     "W": float(stat), "p": float(p)}
        print(f"  k={k:>2}  n={len(sub)}  n_nonzero={len(nz)}  "
              f"mean_delta={np.mean(diffs):+.3f}  W={stat:.2f}  p={p:.4g}")

    # ---- 2. Per-category dilution with bootstrap CI ----
    print("\n=== Per-category dilution (k=10, bootstrap CI) ===")
    k10 = [r for r in rows if r["k"] == 10]
    rng = np.random.default_rng(0)
    cat_stats = {}
    for c in sorted(set(r["category"] for r in k10)):
        cv = [r["delta"] for r in k10 if r["category"] == c]
        if not cv: continue
        boot = np.array(
            [np.mean(rng.choice(cv, size=len(cv), replace=True))
             for _ in range(5000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        cat_stats[c] = {
            "n": int(len(cv)),
            "mean": float(np.mean(cv)),
            "ci_lo": float(lo), "ci_hi": float(hi),
            "density": int(AGENT_CHUNK_COUNTS[c]),
        }
        print(f"  {c:>20}  n={len(cv):>3}  delta={np.mean(cv):+.3f}  "
              f"95%CI=[{lo:+.3f},{hi:+.3f}]  N_c={AGENT_CHUNK_COUNTS[c]}")

    # ---- 3. Per-query correlations ----
    deltas = np.array([r["delta"] for r in k10])
    logd   = np.array([r["log_density"] for r in k10])
    if deltas.std() > 0 and logd.std() > 0:
        pr, ppr = pearsonr(logd, deltas)
        sr, psr = spearmanr(logd, deltas)
    else:
        pr = ppr = sr = psr = float("nan")
    print(f"\nPer-query (n={len(deltas)})  "
          f"Pearson(log_density, delta) = {pr:+.3f}  p={ppr:.4g}; "
          f"Spearman = {sr:+.3f}  p={psr:.4g}")

    # ---- Plot ----
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "legend.fontsize": 7.5, "pdf.fonttype": 42,
    })
    cats_sorted = sorted(cat_stats.keys(), key=lambda c: cat_stats[c]["density"])
    xs = [cat_stats[c]["density"] for c in cats_sorted]
    means = [cat_stats[c]["mean"] for c in cats_sorted]
    los = [cat_stats[c]["mean"] - cat_stats[c]["ci_lo"] for c in cats_sorted]
    his = [cat_stats[c]["ci_hi"] - cat_stats[c]["mean"] for c in cats_sorted]
    fig, ax = plt.subplots(figsize=(3.4, 2.6), constrained_layout=True)
    ax.errorbar(xs, means, yerr=[los, his],
                fmt="o", color="#1f77b4",
                markeredgecolor="#0a3a66", markeredgewidth=0.6,
                ecolor="#1f77b4", elinewidth=1.0, capsize=3, zorder=3)
    nice = lambda c: c.replace("_", " ").title()
    for x, y, c in zip(xs, means, cats_sorted):
        ax.annotate(nice(c), (x, y), xytext=(6, 3),
                    textcoords="offset points",
                    fontsize=7, color="#222")
    ax.axhline(0, color="#888", linewidth=0.6, linestyle=":", zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel("Chunks per category (log scale)")
    ax.set_ylabel(r"Per-query $\Delta = \text{hit}_{\text{scoped}} - \text{hit}_{\text{global}}$ ($k{=}10$)")
    ax.set_ylim(-0.15, 0.55)
    ax.grid(True, linestyle=":", linewidth=0.5, color="#ccc", alpha=0.6)
    ax.set_axisbelow(True)
    if not np.isnan(pr):
        n_now = len(deltas)
        label = (f"per-query Pearson r = {pr:+.2f}, p = {ppr:.2g}\n"
                 f"n = {n_now} (was n = 8)")
        ax.text(0.97, 0.05, label,
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="#bbb", lw=0.5))
    for ext in ("pdf", "png"):
        out = FIG_DIR / f"perquery_dilution.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"[save] {out}")
    plt.close(fig)

    (FIG_DIR / "perquery_dilution_stats.json").write_text(json.dumps({
        "wilcoxon_per_k": wilcox,
        "category_stats": cat_stats,
        "global_pearson": {"r": float(pr) if not np.isnan(pr) else None,
                           "p": float(ppr) if not np.isnan(ppr) else None,
                           "n": int(len(deltas))},
        "global_spearman": {"r": float(sr) if not np.isnan(sr) else None,
                            "p": float(psr) if not np.isnan(psr) else None,
                            "n": int(len(deltas))},
        "n_queries_used": len(set(r["qid"] for r in rows)),
    }, indent=2))
    print(f"[save] {FIG_DIR/'perquery_dilution_stats.json'}")


if __name__ == "__main__":
    main()
