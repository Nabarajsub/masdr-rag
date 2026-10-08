"""Retrieval-source confusion on Composite-9, all nine sources, canonical run
(results/composite_qwen_promptctl_singlecall.jsonl; same run as Table 4)."""
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

R = Path("<DATA_ROOT>")
OUT = Path("<DATA_ROOT>")
SOURCES = ["helpdesk", "confluence", "stackoverflow", "gmail", "docs", "slack", "github", "jira", "reports"]
PRETTY = {"helpdesk": "Helpdesk", "confluence": "Confluence", "stackoverflow": "StackOverflow", "gmail": "Gmail",
          "docs": "Docs", "slack": "Slack", "github": "GitHub", "jira": "Jira", "reports": "Reports"}
SYSTEMS = [("monolithic", "(a) Monolithic"), ("regex_scoped", "(b) Regex-Scoped"),
           ("hybrid_routed", "(c) Hybrid-Routed (hard)"), ("soft_scoped", "(d) Soft-Scoped")]

recs = [json.loads(l) for l in open(R / "composite_qwen_promptctl_singlecall.jsonl")]

def matrix(system):
    by = defaultdict(Counter)
    for r in recs:
        if r.get("system") != system or "error" in r or r.get("category") not in SOURCES:
            continue
        for s in r.get("chunk_sources") or []:
            if s in SOURCES:
                by[r["category"]][s] += 1
    M = np.zeros((9, 9))
    for i, g in enumerate(SOURCES):
        tot = sum(by[g].values())
        for j, s in enumerate(SOURCES):
            M[i, j] = by[g][s] / tot if tot else 0
    return M

plt.rcParams.update({"font.family": "serif", "font.size": 9, "pdf.fonttype": 42})
cmap = LinearSegmentedColormap.from_list("b", ["#ffffff", "#cfe1f2", "#6baed6", "#2171b5", "#08306b"])
fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.0), constrained_layout=True)
labels = [PRETTY[s] for s in SOURCES]
for ax, (sysname, title) in zip(axes.flat, SYSTEMS):
    M = matrix(sysname)
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1)
    for i in range(9):
        for j in range(9):
            if M[i, j] >= 0.05:
                ax.text(j, i, f"{M[i,j]:.2f}".lstrip("0"), ha="center", va="center", fontsize=6.5,
                        color="white" if M[i, j] > 0.55 else "#111")
    ax.set_xticks(range(9), labels, rotation=50, ha="right", fontsize=7.5)
    ax.set_yticks(range(9), labels, fontsize=7.5)
    ax.set_title(f"{title}\ndiagonal avg = {np.mean(np.diag(M)):.2f}", fontsize=9.5, fontweight="bold")
    ax.set_xlabel("Retrieved chunk source", fontsize=8)
    ax.set_ylabel("Query gold source", fontsize=8)
    print(sysname, round(float(np.mean(np.diag(M))), 3))
cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02)
cb.set_label("Share of retrieved chunks", fontsize=8)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"retrieval_confusion_v2.{ext}", dpi=300, bbox_inches="tight")
print("saved")
