"""S-1 + N-1: per-query dilution analysis with density debiasing.

The original category-aggregate analysis (n=8) had Spearman = -0.60,
p = 0.12 (not significant). This lifts to per-query observations using
the WYDOT test-suite's `relevant_title` + `chunk_series` (corrected
matcher --- earlier attempt used chunk-id matching which produced too
few non-zero observations).

Defines hit_at_k for monolithic vs regex_scoped:
    hit_s,k(q) = 1 if relevant_title in chunk_series[:k] else 0
    delta_q,k = hit_scoped,k(q) - hit_global,k(q)

Then:
  * Wilcoxon paired test at each k
  * Per-category bootstrap CIs on delta
  * Per-query Pearson + Spearman against log_density(category)
  * **N-1**: density-debias regression
      delta = beta_0 + beta_1 * log_density + epsilon
    and report the slope's significance.

Outputs:
    figures/perquery_dilution_v2.pdf
    figures/perquery_dilution_v2.json
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, pearsonr, spearmanr, linregress

ROOT = Path("<DATA_ROOT>")
FIG_DIR = Path("<DATA_ROOT>")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Pull the 4-shard WYDOT runs (qwen + gemini embedder) — these are the
# canonical retrieval traces used for the paper.
INFILES = sorted(glob.glob(str(ROOT / "wydot_qwen_gemini_*_shard*.judged.jsonl")))

# Per-agent chunk counts on WYDOT (Table 3).
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


def hit_at_k(chunk_series, gold_title: str, k: int) -> int:
    if not gold_title or not chunk_series:
        return 0
    target = gold_title.strip().lower()
    head = [(s or "").strip().lower() for s in chunk_series[:k]]
    return 1 if target in head else 0


def main() -> None:
    print(f"[load] {len(INFILES)} shards", flush=True)
    by_q = defaultdict(dict)
    for f in INFILES:
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = r.get("query_id")
            sys_name = r.get("system")
            if qid and sys_name:
                by_q[qid][sys_name] = r
    print(f"[load] {len(by_q)} unique queries", flush=True)

    rows = []
    for qid, by_sys in by_q.items():
        mono = by_sys.get("monolithic")
        scop = by_sys.get("regex_scoped")
        if not mono or not scop:
            continue
        gold = mono.get("relevant_title")
        cat = mono.get("category")
        if cat not in AGENT_CHUNK_COUNTS or not gold:
            continue
        for k in (3, 5, 10, 15):
            p_g = hit_at_k(mono.get("chunk_series") or [], gold, k)
            p_s = hit_at_k(scop.get("chunk_series") or [], gold, k)
            rows.append({
                "qid": qid, "category": cat, "k": k,
                "p_global": p_g, "p_scoped": p_s, "delta": p_s - p_g,
                "log_density": float(np.log10(AGENT_CHUNK_COUNTS[cat])),
            })
    print(f"per-query observations: {len(rows)}", flush=True)
    print(f"unique queries used:   {len(set(r['qid'] for r in rows))}", flush=True)

    # 1. Wilcoxon paired test
    print("\n=== Wilcoxon (paired) hit_scoped vs hit_global ===")
    wilcox = {}
    for k in (3, 5, 10, 15):
        sub = [r for r in rows if r["k"] == k]
        diffs = [r["delta"] for r in sub]
        nz = [d for d in diffs if d != 0]
        if len(nz) < 5:
            print(f"  k={k}: too few non-zero ({len(nz)})")
            wilcox[k] = {"n": len(sub), "n_nz": len(nz)}
            continue
        stat, p = wilcoxon(nz, alternative="two-sided", zero_method="pratt")
        wilcox[k] = {"n": int(len(sub)), "n_nz": int(len(nz)),
                     "mean_delta": float(np.mean(diffs)),
                     "W": float(stat), "p": float(p)}
        print(f"  k={k:>2}  n={len(sub)}  n_nz={len(nz)}  "
              f"mean_delta={np.mean(diffs):+.3f}  W={stat:.2f}  p={p:.4g}")

    # 2. Per-category bootstrap (k=10)
    print("\n=== Per-category dilution (k=10, bootstrap 95% CI) ===")
    k10 = [r for r in rows if r["k"] == 10]
    rng = np.random.default_rng(0)
    cat_stats = {}
    for c in sorted(set(r["category"] for r in k10)):
        cv = [r["delta"] for r in k10 if r["category"] == c]
        if not cv:
            continue
        boot = np.array([np.mean(rng.choice(cv, size=len(cv), replace=True))
                         for _ in range(5000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        cat_stats[c] = {"n": int(len(cv)), "mean": float(np.mean(cv)),
                        "ci_lo": float(lo), "ci_hi": float(hi),
                        "density": int(AGENT_CHUNK_COUNTS[c])}
        print(f"  {c:>20}  n={len(cv):>3}  "
              f"delta={np.mean(cv):+.3f}  95%CI=[{lo:+.3f},{hi:+.3f}]  N_c={AGENT_CHUNK_COUNTS[c]}")

    # 3. Per-query correlations
    deltas = np.array([r["delta"] for r in k10])
    logd = np.array([r["log_density"] for r in k10])
    if deltas.std() > 0 and logd.std() > 0:
        pr, ppr = pearsonr(logd, deltas)
        sr, psr = spearmanr(logd, deltas)
    else:
        pr = ppr = sr = psr = float("nan")
    print(f"\nPer-query (n={len(deltas)})  Pearson(log_density, delta) = {pr:+.3f}  p={ppr:.4g}")
    print(f"                              Spearman = {sr:+.3f}  p={psr:.4g}")

    # 4. N-1 density-debias regression
    print("\n=== N-1: linear regression delta ~ log_density ===")
    if deltas.std() > 0 and logd.std() > 0:
        slope, intercept, r, p, se = linregress(logd, deltas)
        print(f"  slope = {slope:+.4f} ± {se:.4f} (SE)   r={r:+.3f}   p={p:.4g}")
        print(f"  intercept = {intercept:+.4f}")
        # Effect size: drop in dilution per decade of N_c
        print(f"  Effect: each 10x increase in N_c → "
              f"dilution shifts by {slope:+.3f} hits/query at k=10.")
        debias = {"slope": float(slope), "slope_se": float(se),
                  "intercept": float(intercept),
                  "r": float(r), "p": float(p)}
    else:
        debias = {"slope": None}

    # 5. Plot
    plt.rcParams.update({"font.family": "serif", "font.size": 9,
                         "axes.titlesize": 10, "axes.labelsize": 9,
                         "legend.fontsize": 7.5, "pdf.fonttype": 42})
    cats_sorted = sorted(cat_stats.keys(), key=lambda c: cat_stats[c]["density"])
    xs = [cat_stats[c]["density"] for c in cats_sorted]
    means = [cat_stats[c]["mean"] for c in cats_sorted]
    los = [cat_stats[c]["mean"] - cat_stats[c]["ci_lo"] for c in cats_sorted]
    his = [cat_stats[c]["ci_hi"] - cat_stats[c]["mean"] for c in cats_sorted]
    fig, ax = plt.subplots(figsize=(3.4, 2.6), constrained_layout=True)
    ax.errorbar(xs, means, yerr=[los, his], fmt="o", color="#1f77b4",
                markeredgecolor="#0a3a66", markeredgewidth=0.6,
                ecolor="#1f77b4", elinewidth=1.0, capsize=3, zorder=3)
    nice = lambda c: c.replace("_", " ").title()
    for x, y, c in zip(xs, means, cats_sorted):
        ax.annotate(nice(c), (x, y), xytext=(6, 3), textcoords="offset points",
                    fontsize=7, color="#222")
    # Overlay regression line
    if debias.get("slope") is not None:
        log_x = np.log10(np.array(xs))
        x_grid = np.linspace(log_x.min() - 0.1, log_x.max() + 0.1, 50)
        y_grid = debias["slope"] * x_grid + debias["intercept"]
        ax.plot(10 ** x_grid, y_grid, "-", color="#d62728", lw=1.2, zorder=2,
                label=f"slope={debias['slope']:+.3f}, p={debias['p']:.2g}")
        ax.legend(loc="upper right", frameon=False, fontsize=7)
    ax.axhline(0, color="#888", linewidth=0.6, linestyle=":", zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel("Chunks per category (log scale)")
    ax.set_ylabel(r"Per-query $\Delta = \text{hit}_{\text{scoped}} - \text{hit}_{\text{global}}$ ($k{=}10$)")
    ax.set_ylim(-0.2, 0.6)
    ax.grid(True, linestyle=":", linewidth=0.5, color="#ccc", alpha=0.6)
    ax.set_axisbelow(True)
    out_pdf = FIG_DIR / "perquery_dilution_v2.pdf"
    out_png = FIG_DIR / "perquery_dilution_v2.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[save] {out_pdf}")

    summary = {
        "n_queries_used": len(set(r["qid"] for r in rows)),
        "n_observations": len(rows),
        "wilcoxon_per_k": wilcox,
        "category_stats": cat_stats,
        "perquery_pearson": {"r": None if np.isnan(pr) else float(pr),
                             "p": None if np.isnan(ppr) else float(ppr),
                             "n": int(len(deltas))},
        "perquery_spearman": {"r": None if np.isnan(sr) else float(sr),
                              "p": None if np.isnan(psr) else float(psr),
                              "n": int(len(deltas))},
        "density_regression": debias,
    }
    (FIG_DIR / "perquery_dilution_v2.json").write_text(json.dumps(summary, indent=2))
    print(f"[save] {FIG_DIR/'perquery_dilution_v2.json'}")


if __name__ == "__main__":
    main()
