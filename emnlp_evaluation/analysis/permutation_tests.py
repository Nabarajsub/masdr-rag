"""
Paired permutation tests + bootstrap CIs.

For every pair (system_a, system_b) within the same (llm, embedder, benchmark),
compute:
  * Δ correctness  (mean(a) - mean(b))
  * permutation-test p-value (10k shuffles, two-sided)
  * 95% bootstrap CI on the difference

Output: emnlp_evaluation/results/permutation_tests.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List


def _load(glob_pat: str):
    rows = []
    for p in glob.glob(glob_pat):
        for line in open(p):
            try:
                r = json.loads(line)
                if "error" not in r:
                    rows.append(r)
            except json.JSONDecodeError:
                continue
    return rows


def _by_query(rows, metric: str):
    """Index rows as (bench, llm, embedder, system) -> {query_id -> value}."""
    idx = defaultdict(dict)
    for r in rows:
        v = r.get(metric)
        if v is None:
            continue
        key = (r.get("benchmark", "wydot"), r.get("llm", "?"),
               r.get("embedder", "?"), r.get("system", "?"))
        idx[key][r["query_id"]] = float(v)
    return idx


def _perm_test(paired: List[tuple], n_perm: int = 10_000, seed: int = 0) -> tuple:
    """Two-sided paired permutation test on the difference."""
    rng = random.Random(seed)
    diffs = [a - b for a, b in paired]
    if not diffs:
        return float("nan"), float("nan"), float("nan")
    obs = sum(diffs) / len(diffs)
    cnt = 0
    for _ in range(n_perm):
        s = 0.0
        for d in diffs:
            s += d if rng.random() < 0.5 else -d
        s /= len(diffs)
        if abs(s) >= abs(obs) - 1e-12:
            cnt += 1
    p = cnt / n_perm
    # 95% bootstrap CI on the mean diff
    n_boot = 2000
    boots = []
    for _ in range(n_boot):
        sample = [rng.choice(diffs) for _ in range(len(diffs))]
        boots.append(sum(sample) / len(sample))
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot)]
    return obs, p, (lo, hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="emnlp_evaluation/results/*.judged.jsonl")
    ap.add_argument("--metric", default="correctness",
                    choices=("correctness", "faithfulness"))
    ap.add_argument("--out", default="emnlp_evaluation/results/permutation_tests.csv")
    ap.add_argument("--n-perm", type=int, default=10_000)
    args = ap.parse_args()

    rows = _load(args.inputs)
    if not rows:
        raise SystemExit(f"no rows match {args.inputs}")
    by_q = _by_query(rows, args.metric)

    # Group keys by (benchmark, llm, embedder) so we only compare systems
    # evaluated under the same conditions.
    groups = defaultdict(list)
    for key in by_q:
        groups[key[:3]].append(key)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "llm", "embedder",
                    "system_a", "system_b", "n_shared",
                    "mean_a", "mean_b", "delta", "p_value",
                    "ci_lo", "ci_hi"])
        for (bench, llm, emb), keys in sorted(groups.items()):
            for a, b in combinations(keys, 2):
                sa, sb = a[3], b[3]
                qa = by_q[a]; qb = by_q[b]
                shared = sorted(set(qa) & set(qb))
                if len(shared) < 5:
                    continue
                paired = [(qa[q], qb[q]) for q in shared]
                obs, p, ci = _perm_test(paired, n_perm=args.n_perm)
                mean_a = sum(x for x, _ in paired) / len(paired)
                mean_b = sum(y for _, y in paired) / len(paired)
                if isinstance(ci, tuple):
                    lo, hi = ci
                else:
                    lo, hi = float("nan"), float("nan")
                w.writerow([bench, llm, emb, sa, sb, len(shared),
                            f"{mean_a:.3f}", f"{mean_b:.3f}",
                            f"{obs:+.3f}", f"{p:.4f}",
                            f"{lo:+.3f}", f"{hi:+.3f}"])

    print(f"[perm] wrote {args.out}")


if __name__ == "__main__":
    main()
