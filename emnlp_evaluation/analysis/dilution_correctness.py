"""S-1 v2: per-query dilution using correctness deltas (judge labels).

The relevant_title-based matcher produces too-few non-zero observations
because only ~32 of 200 queries carry an explicit gold title. We pivot
to the metric the reviewer actually cares about: does scoping help
correctness more in dense categories than in sparse ones?

For each query q, compute
    delta_q = corr_scoped(q) - corr_global(q)
where both are 0/1 from the LLM-as-judge.

Then regress delta on log_density(category) per query (N >> 8).

Inputs (post re-judge with chunk DB):
    wydot_qwen_gemini_*_shardNofN.judged.jsonl

Outputs:
    figures/perquery_dilution_corr.{pdf,png,json}
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

INFILES = sorted(glob.glob(str(ROOT / "wydot_qwen_gemini_*_shard*.judged.jsonl")))

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


def main():
    print(f"[load] {len(INFILES)} shards", flush=True)
    by_q = defaultdict(dict)
    seen_keys = set()
    for f in INFILES:
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = r.get("query_id")
            sys = r.get("system")
            if not (qid and sys):
                continue
            key = (qid, sys)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            by_q[qid][sys] = r
    print(f"[load] {len(by_q)} unique queries", flush=True)

    rows = []
    for qid, by_sys in by_q.items():
        mono = by_sys.get("monolithic")
        scop = by_sys.get("regex_scoped") or by_sys.get("hybrid_routed")
        if not mono or not scop:
            continue
        cat = mono.get("category")
        if cat not in AGENT_CHUNK_COUNTS:
            continue
        corr_g = mono.get("correctness")
        corr_s = scop.get("correctness")
        if corr_g is None or corr_s is None:
            continue
        rows.append({
            "qid": qid, "category": cat,
            "corr_global": int(corr_g), "corr_scoped": int(corr_s),
            "delta": int(corr_s) - int(corr_g),
            "log_density": float(np.log10(AGENT_CHUNK_COUNTS[cat])),
        })

    n = len(rows)
    print(f"observations: {n}  (queries with both monolithic and a scoped variant judged)")
    if n < 30:
        print("Not enough data — aborting.")
        return

    diffs = np.array([r["delta"] for r in rows])
    nz = diffs[diffs != 0]
    mean = float(diffs.mean())
    print(f"\n=== Paired Wilcoxon: corr_scoped vs corr_global ===")
    if len(nz) >= 5:
        stat, p = wilcoxon(nz, alternative="two-sided", zero_method="pratt")
        print(f"  n={n}  n_nz={len(nz)}  mean_delta={mean:+.3f}  W={stat:.2f}  p={p:.4g}")
        wilcox = {"n": int(n), "n_nz": int(len(nz)),
                  "mean_delta": mean, "W": float(stat), "p": float(p)}
    else:
        wilcox = {"n": int(n), "n_nz": int(len(nz))}
        print(f"  too few non-zero ({len(nz)})")

    print(f"\n=== Per-category dilution (bootstrap 95% CI) ===")
    rng = np.random.default_rng(0)
    cat_stats = {}
    for c in sorted(set(r["category"] for r in rows)):
        cv = np.array([r["delta"] for r in rows if r["category"] == c], dtype=float)
        if len(cv) < 3:
            continue
        boot = np.array([rng.choice(cv, size=len(cv), replace=True).mean()
                         for _ in range(5000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        cat_stats[c] = {"n": int(len(cv)), "mean": float(cv.mean()),
                        "ci_lo": float(lo), "ci_hi": float(hi),
                        "density": int(AGENT_CHUNK_COUNTS[c])}
        print(f"  {c:>20}  n={len(cv):>3}  delta={cv.mean():+.3f}  "
              f"95%CI=[{lo:+.3f},{hi:+.3f}]  N_c={AGENT_CHUNK_COUNTS[c]}")

    logd = np.array([r["log_density"] for r in rows], dtype=float)
    if diffs.std() > 0 and logd.std() > 0:
        pr, ppr = pearsonr(logd, diffs)
        sr, psr = spearmanr(logd, diffs)
    else:
        pr = ppr = sr = psr = float("nan")
    print(f"\nPer-query Pearson(log_density, delta) = {pr:+.3f}  p={ppr:.4g}")
    print(f"Per-query Spearman = {sr:+.3f}  p={psr:.4g}")

    print("\n=== N-1: linear regression delta ~ log_density ===")
    if diffs.std() > 0 and logd.std() > 0:
        slope, intercept, r, p, se = linregress(logd, diffs)
        print(f"  slope = {slope:+.4f} ± {se:.4f}   r={r:+.3f}   p={p:.4g}")
        print(f"  intercept = {intercept:+.4f}")
        debias = {"slope": float(slope), "slope_se": float(se),
                  "intercept": float(intercept),
                  "r": float(r), "p": float(p)}
    else:
        debias = {}

    # Plot per-category
    plt.rcParams.update({"font.family": "serif", "font.size": 9,
                         "pdf.fonttype": 42})
    cats = sorted(cat_stats.keys(), key=lambda c: cat_stats[c]["density"])
    xs = [cat_stats[c]["density"] for c in cats]
    means = [cat_stats[c]["mean"] for c in cats]
    los = [cat_stats[c]["mean"] - cat_stats[c]["ci_lo"] for c in cats]
    his = [cat_stats[c]["ci_hi"] - cat_stats[c]["mean"] for c in cats]
    fig, ax = plt.subplots(figsize=(3.4, 2.6), constrained_layout=True)
    ax.errorbar(xs, means, yerr=[los, his], fmt="o", color="#1f77b4",
                ecolor="#1f77b4", elinewidth=1.0, capsize=3, zorder=3)
    for x, y, c in zip(xs, means, cats):
        ax.annotate(c.replace("_", " ").title(),
                    (x, y), xytext=(6, 3), textcoords="offset points",
                    fontsize=7, color="#222")
    if debias.get("slope") is not None:
        log_x = np.log10(np.array(xs))
        x_grid = np.linspace(log_x.min() - 0.1, log_x.max() + 0.1, 50)
        y_grid = debias["slope"] * x_grid + debias["intercept"]
        ax.plot(10 ** x_grid, y_grid, "-", color="#d62728", lw=1.2, zorder=2,
                label=f"slope={debias['slope']:+.3f}, p={debias['p']:.2g}")
        ax.legend(loc="upper right", frameon=False, fontsize=7)
    ax.axhline(0, color="#888", lw=0.6, ls=":", zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel("Chunks per category (log scale)")
    ax.set_ylabel(r"Per-query $\Delta$ correctness (scoped $-$ global)")
    ax.set_ylim(-0.6, 0.6)
    ax.grid(True, ls=":", lw=0.5, color="#ccc", alpha=0.6)
    ax.set_axisbelow(True)
    out_pdf = FIG_DIR / "perquery_dilution_corr.pdf"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "perquery_dilution_corr.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[save] {out_pdf}")

    (FIG_DIR / "perquery_dilution_corr.json").write_text(json.dumps({
        "n_queries": n,
        "wilcoxon": wilcox,
        "category_stats": cat_stats,
        "perquery_pearson": {"r": None if np.isnan(pr) else float(pr),
                             "p": None if np.isnan(ppr) else float(ppr)},
        "perquery_spearman": {"r": None if np.isnan(sr) else float(sr),
                              "p": None if np.isnan(psr) else float(psr)},
        "density_regression": debias,
    }, indent=2))


if __name__ == "__main__":
    main()
