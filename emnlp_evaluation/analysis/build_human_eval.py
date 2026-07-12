"""S-4: build a 50-query human-evaluation sheet.

Stratified sample of 50 WYDOT queries (10 per query_type, capped),
with each query showing the 4 system answers in a shuffled,
system-anonymous order. Output is a CSV the annotator fills in:

    query_id, system_alias, answer, faith_rate, corr_rate, hallu_flag, notes

The mapping system_alias -> real system is saved to a sidecar JSON so
the annotator stays blind during rating.
"""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path("<DATA_ROOT>")
RESULTS = ROOT / "emnlp_evaluation/results"
OUT_DIR = ROOT / "emnlp_evaluation/human_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEMS = ("monolithic", "regex_scoped", "hybrid_routed", "masdr_rag")
SOURCE_FILE = RESULTS / "wydot_qwen_bge_m3_baselines.judged.jsonl"


def main(seed: int = 1, n_per_type: int = 10) -> None:
    by_q: dict[str, dict] = defaultdict(dict)
    for line in open(SOURCE_FILE):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "error" in r:
            continue
        qid = r.get("query_id")
        sys = r.get("system")
        if qid and sys in SYSTEMS:
            by_q[qid][sys] = r

    qids_full = [qid for qid, d in by_q.items() if all(s in d for s in SYSTEMS)]
    print(f"[sample] {len(qids_full)} queries have all 4 systems")
    # Stratify by query_type
    by_type: dict[str, list[str]] = defaultdict(list)
    for qid in qids_full:
        qt = (next(iter(by_q[qid].values())).get("query_type") or "other")
        by_type[qt].append(qid)

    rng = random.Random(seed)
    sample: list[str] = []
    for qt, qids in sorted(by_type.items()):
        rng.shuffle(qids)
        take = min(n_per_type, len(qids))
        sample.extend(qids[:take])
        print(f"  {qt:<22} pool={len(qids):>3} take={take}")
        if len(sample) >= 50:
            break
    sample = sample[:50]
    print(f"[sample] final n={len(sample)}")

    rows = []
    alias_map: dict[tuple[str, str], str] = {}
    for qid in sample:
        d = by_q[qid]
        order = list(SYSTEMS)
        rng.shuffle(order)
        for i, sys in enumerate(order):
            alias = f"S{chr(ord('A')+i)}"
            alias_map[(qid, alias)] = sys
            r = d[sys]
            rows.append({
                "query_id": qid,
                "query": r.get("query", ""),
                "reference": r.get("reference_answer", ""),
                "category": r.get("category", ""),
                "system_alias": alias,
                "answer": r.get("answer", ""),
                "n_chunks": r.get("n_chunks", 0),
                "judge_faith": r.get("faithfulness", ""),
                "judge_corr": r.get("correctness", ""),
                "human_faith_0to3": "",
                "human_corr_0to3": "",
                "hallucination_flag": "",
                "notes": "",
            })

    csv_path = OUT_DIR / "human_eval_50.csv"
    with csv_path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[wrote] {csv_path} ({len(rows)} rows)")

    key_path = OUT_DIR / "human_eval_50_key.json"
    key_path.write_text(json.dumps(
        {f"{qid}__{alias}": sys for (qid, alias), sys in alias_map.items()},
        indent=2))
    print(f"[wrote] {key_path}")


if __name__ == "__main__":
    main()
