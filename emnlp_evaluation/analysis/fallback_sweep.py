"""Confidence-thresholded router fallback sweep (reviewer RBJc W1).

Composes two existing judged runs: for each query, the fallback system
answers with R2-Routed when the R2 classifier's top-1 probability >= tau,
and with Monolithic (global retrieval) otherwise. Because both runs are
already judged per query, the sweep needs no new generation — only the
router confidences.

By construction the tau -> 1 limit is exactly Monolithic, so a misroute can
never leave the composed system below the unscoped baseline.

Usage:
    python -m emnlp_evaluation.analysis.fallback_sweep
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO.parent))

RESULTS = _REPO / "emnlp_evaluation" / "results"
R2_FILE = RESULTS / "wydot_qwen_bge_m3_r2_routed.judged.jsonl"
BASE_FILE = RESULTS / "wydot_qwen_bge_m3_baselines.judged.jsonl"
OUT = RESULTS / "fallback_sweep.json"

TAUS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]


def _load(path: Path, system: str) -> dict:
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("system") == system and "error" not in r:
            out[r["query_id"]] = r
    return out


def main() -> None:
    from emnlp_evaluation.agents.r2_router import R2Router

    r2 = _load(R2_FILE, "r2_routed")
    mono = _load(BASE_FILE, "monolithic")
    shared = sorted(set(r2) & set(mono))
    print(f"[fallback] {len(r2)} r2 rows, {len(mono)} mono rows, {len(shared)} shared")

    router = R2Router.get()
    conf = {}
    for i, qid in enumerate(shared):
        _, p = router.route_with_confidence(r2[qid]["query"])
        conf[qid] = p
        if (i + 1) % 50 == 0:
            print(f"[fallback] {i+1} confidences", flush=True)

    rows = []
    for tau in TAUS:
        corr = faith = fb = 0
        for qid in shared:
            rec = r2[qid] if conf[qid] >= tau else mono[qid]
            if conf[qid] < tau:
                fb += 1
            corr += rec.get("correctness", 0)
            faith += rec.get("faithfulness", 0)
        n = len(shared)
        rows.append({"tau": tau, "n": n, "fallback_rate": fb / n,
                     "correctness": corr / n, "faithfulness": faith / n})
        print(f"tau={tau:4.2f} fallback={fb/n:5.1%} corr={corr/n:.3f} faith={faith/n:.3f}")

    OUT.write_text(json.dumps({"taus": rows, "confidences": conf}, indent=1))
    print(f"[fallback] wrote {OUT}")


if __name__ == "__main__":
    main()
