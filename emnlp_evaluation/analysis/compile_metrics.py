"""
Aggregator: run every analysis script and dump a single results.json the
paper can consume directly. Also regenerates the Pareto plot.

Outputs (under emnlp_evaluation/results/):
  summary.csv                — per-system aggregates (from latency_pareto)
  permutation_tests.csv      — pairwise significance (from permutation_tests)
  dilution_table.csv         — per-category/source dilution
  pareto.png                 — accuracy vs. latency
  metrics.json               — everything above, joined for paper consumption
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
PYBIN = sys.executable


def _run(*args):
    print("$", " ".join(args), flush=True)
    r = subprocess.run(args, cwd=str(BASE.parent), capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print("STDERR:", r.stderr, file=sys.stderr)
    return r.returncode


def _csv_rows(path: Path):
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    res = BASE / "results"
    res.mkdir(exist_ok=True)

    # 1) Latency / Pareto (judged JSONL only so correctness columns are populated)
    _run(PYBIN, "-m", "emnlp_evaluation.analysis.latency_pareto",
         "--inputs", "results/*.judged.jsonl", "--plot")

    # 2) Permutation tests (correctness then faithfulness)
    _run(PYBIN, "-m", "emnlp_evaluation.analysis.permutation_tests",
         "--inputs", "emnlp_evaluation/results/*.judged.jsonl",
         "--metric", "correctness",
         "--out", "emnlp_evaluation/results/permutation_correctness.csv")
    _run(PYBIN, "-m", "emnlp_evaluation.analysis.permutation_tests",
         "--inputs", "emnlp_evaluation/results/*.judged.jsonl",
         "--metric", "faithfulness",
         "--out", "emnlp_evaluation/results/permutation_faithfulness.csv")

    # 3) Dilution analysis
    _run(PYBIN, "-m", "emnlp_evaluation.analysis.dilution_analysis",
         "--inputs", "results/*.judged.jsonl")

    # 4) Bundle into metrics.json for the paper
    bundle = {
        "summary":     _csv_rows(res / "summary.csv"),
        "perm_correct": _csv_rows(res / "permutation_correctness.csv"),
        "perm_faith":   _csv_rows(res / "permutation_faithfulness.csv"),
    }
    out = res / "metrics.json"
    with open(out, "w") as f:
        json.dump(bundle, f, indent=2)
    print(f"[compile_metrics] wrote {out}")


if __name__ == "__main__":
    main()
