"""Token-and-call efficiency figure.

Two stacked panels on WYDOT (Qwen-2.5-7B and Llama-3-8B stacks, Gemini
judge). Left axis: mean prompt + completion tokens per query (stacked).
Right axis: mean LLM-call count per query (line overlay).

Highlights:
  * Monolithic + Regex-Scoped: 1 call, smallest token bill.
  * Hybrid-Routed: 1-2 calls (router + synth).
  * MASDR-RAG: ~2 calls, ~+15% tokens.
  * Custom ReAct: explodes on Llama (5.5 calls, ~39k tokens).
"""
from __future__ import annotations
import json, glob
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT = Path("<DATA_ROOT>")
OUT = Path("<DATA_ROOT>")
OUT.mkdir(parents=True, exist_ok=True)

SYSTEMS = [
    ("Monolithic",     "monolithic"),
    ("Regex-Scoped",   "regex_scoped"),
    ("Hybrid-Routed",  "hybrid_routed"),
    ("MASDR-RAG",      "masdr_rag"),
    ("Custom ReAct",   "react"),
]
PANELS = [
    ("Qwen-2.5-7B backbone",  f"{ROOT}/wydot_qwen_gemini_*_shard*.judged.jsonl"),
    ("Llama-3-8B backbone",   f"{ROOT}/wydot_llama_gemini_*_shard*.judged.jsonl"),
]


def aggregate(pattern: str, sysname: str):
    prompt, completion, calls = [], [], []
    for f in glob.glob(pattern):
        with open(f) as fh:
            for line in fh:
                try: r = json.loads(line)
                except: continue
                if r.get("system") != sysname: continue
                if r.get("prompt_tokens") is not None:
                    prompt.append(int(r.get("prompt_tokens") or 0))
                if r.get("completion_tokens") is not None:
                    completion.append(int(r.get("completion_tokens") or 0))
                c = r.get("llm_calls")
                if isinstance(c, (int, float)): calls.append(int(c))
    if not prompt: return None
    return {
        "n":         len(prompt),
        "prompt":    float(np.mean(prompt)),
        "completion":float(np.mean(completion)),
        "total":     float(np.mean(prompt)) + float(np.mean(completion)),
        "calls":     float(np.mean(calls)) if calls else 0.0,
    }


def main():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "legend.fontsize": 7.5, "pdf.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), constrained_layout=True)
    bw = 0.62
    x = np.arange(len(SYSTEMS))

    for ax, (title, pat) in zip(axes, PANELS):
        rows = [aggregate(pat, s[1]) for s in SYSTEMS]
        prompts = [r["prompt"] if r else 0 for r in rows]
        comps   = [r["completion"] if r else 0 for r in rows]
        calls   = [r["calls"] if r else 0 for r in rows]
        for s, r in zip(SYSTEMS, rows):
            n = r["n"] if r else 0
            print(f"  {title:>20} | {s[0]:>14} | n={n:>4} | prompt={r['prompt'] if r else 0:7.0f} "
                  f"| comp={r['completion'] if r else 0:6.0f} | calls={r['calls'] if r else 0:.2f}")

        b1 = ax.bar(x, prompts, bw, color="#4c9be8", edgecolor="black",
                    linewidth=0.4, label="Prompt tokens")
        b2 = ax.bar(x, comps, bw, bottom=prompts, color="#f2a93a",
                    edgecolor="black", linewidth=0.4, label="Completion tokens")
        # value labels (total tokens)
        totals = [p + c for p, c in zip(prompts, comps)]
        ymax = max(totals) * 1.18
        for xi, t in zip(x, totals):
            if t > 0:
                ax.text(xi, t + ymax * 0.02, f"{t/1000:.1f}k",
                        ha="center", va="bottom", fontsize=6.8, color="#222")

        ax.set_xticks(x, [s[0] for s in SYSTEMS], rotation=22, ha="right", fontsize=7.5)
        ax.set_ylim(0, ymax)
        ax.set_title(title, fontweight="bold")
        if ax is axes[0]:
            ax.set_ylabel("Tokens per query")
        ax.grid(True, axis="y", linestyle=":", linewidth=0.5,
                color="#ccc", alpha=0.6)
        ax.set_axisbelow(True)

        ax2 = ax.twinx()
        ax2.plot(x, calls, color="#d62728", marker="o", linewidth=1.5,
                 markersize=5, markeredgecolor="black", markeredgewidth=0.4,
                 label="LLM calls (mean)", zorder=5)
        for xi, cv in zip(x, calls):
            ax2.text(xi + 0.18, cv, f"{cv:.1f}",
                     fontsize=6.5, color="#a01e1e", va="center")
        ax2.set_ylim(0, max(6.5, max(calls) * 1.25))
        if ax is axes[-1]:
            ax2.set_ylabel("LLM calls per query", color="#a01e1e")
        ax2.tick_params(axis="y", labelcolor="#a01e1e", labelsize=7)
        ax2.spines["right"].set_color("#a01e1e")

    # Shared legend (top)
    legend_handles = [
        Patch(facecolor="#4c9be8", edgecolor="black", label="Prompt tokens"),
        Patch(facecolor="#f2a93a", edgecolor="black", label="Completion tokens"),
        Line2D([0], [0], color="#d62728", marker="o", markersize=5,
               markeredgecolor="black", markeredgewidth=0.4,
               label="LLM calls per query"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)

    for ext in ("pdf", "png"):
        out = OUT / f"efficiency_tokens_calls.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"[save] {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
