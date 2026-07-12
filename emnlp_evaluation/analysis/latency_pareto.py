"""
Per-system latency / token / accuracy table + optional Pareto plot.

Reads JSONL outputs from runners/ and (optionally) the .judged.jsonl outputs
from analysis/judge.py. Produces:
  * Markdown table (printed to stdout)
  * results/summary.csv
  * results/pareto.png (if --plot)
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def _load_records(glob_pattern: str):
    rows = []
    for path in glob.glob(glob_pattern):
        with open(path) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def _pct(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def summarize(rows):
    """Group by (llm, embedder, system); compute aggregates."""
    groups = defaultdict(list)
    for r in rows:
        if "error" in r:
            continue
        key = (r.get("llm", "?"), r.get("embedder", "?"), r.get("system", "?"))
        groups[key].append(r)

    summary = []
    for (llm, emb, sys_name), recs in sorted(groups.items()):
        lats = [r.get("wall_time_s", 0) for r in recs]
        tokens = [r.get("prompt_tokens", 0) + r.get("completion_tokens", 0) for r in recs]
        llm_calls = [r.get("llm_calls", 0) for r in recs]
        corr = [r["correctness"] for r in recs if "correctness" in r]
        faith = [r["faithfulness"] for r in recs if "faithfulness" in r]
        summary.append({
            "llm": llm, "embedder": emb, "system": sys_name,
            "n": len(recs),
            "latency_p50": _pct(lats, 50),
            "latency_p95": _pct(lats, 95),
            "tokens_mean": statistics.mean(tokens) if tokens else 0,
            "llm_calls_mean": statistics.mean(llm_calls) if llm_calls else 0,
            "correctness": statistics.mean(corr) if corr else float("nan"),
            "faithfulness": statistics.mean(faith) if faith else float("nan"),
        })
    return summary


def print_markdown(summary):
    cols = ["llm", "embedder", "system", "n", "latency_p50", "latency_p95",
            "tokens_mean", "llm_calls_mean", "correctness", "faithfulness"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for row in summary:
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:.3f}" if not math.isnan(v) else "—")
            else:
                cells.append(str(v))
        print("| " + " | ".join(cells) + " |")


def save_csv(summary, path: Path):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"[latency_pareto] wrote {path}")


def plot_pareto(summary, out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot."); return

    fig, ax = plt.subplots(figsize=(7, 5))
    for row in summary:
        x = row["latency_p50"]
        y = row["correctness"] if not math.isnan(row.get("correctness", float("nan"))) else 0
        label = f"{row['system']} ({row['llm']})"
        ax.scatter(x, y, s=80)
        ax.annotate(label, (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Latency p50 (s)")
    ax.set_ylabel("Correctness")
    ax.set_title("Pareto: accuracy vs latency")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[latency_pareto] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="results/*.jsonl",
                    help="Glob (relative to emnlp_evaluation/).")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    base = Path(__file__).resolve().parents[1]
    rows = _load_records(str(base / args.inputs))
    summary = summarize(rows)
    if not summary:
        print("No rows found."); return
    print_markdown(summary)
    save_csv(summary, base / "results" / "summary.csv")
    if args.plot:
        plot_pareto(summary, base / "results" / "pareto.png")


if __name__ == "__main__":
    main()
