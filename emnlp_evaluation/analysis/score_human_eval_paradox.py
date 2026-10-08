#!/usr/bin/env python3
"""Score the paradox human-eval once annotators have filled their CSVs.

Answers the reviewer question: does the precision-faithfulness paradox hold
under HUMAN faithfulness judgment, independent of the LLM judge?

Reads paradox_annotator{1,2}.csv + paradox_key.json and reports:
  1. Mean human faithfulness per system (Monolithic vs MASDR-RAG), paired delta,
     paired-permutation p  -> does the paradox survive human scoring?
  2. Inter-annotator agreement (Pearson r + exact-agreement rate on {1,.5,0}).
  3. Human-vs-LLM agreement (Pearson r) -> is the LLM judge measuring the same
     construct humans are, or just penalizing orchestrated outputs?

Usage:
    python -m emnlp_evaluation.analysis.score_human_eval_paradox
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "human_eval"
rng = np.random.default_rng(0)


def read_annot(path):
    """Read either the website's JSON export (paradox_<who>.json with a
    {ratings:{uid:{faithful:..}}} map) or the legacy CSV sheet."""
    path = Path(path)
    if path.suffix == ".json":
        blob = json.loads(path.read_text())
        return {uid: float(r["faithful"])
                for uid, r in blob.get("ratings", {}).items()
                if r.get("faithful") is not None}
    d = {}
    for row in csv.DictReader(open(path)):
        v = (row.get("faithful_1_0.5_0") or "").strip()
        if v == "":
            continue
        try:
            d[row["uid"]] = float(v)
        except ValueError:
            pass
    return d


def find_annot_files():
    """Prefer website JSON exports (paradox_*.json, excluding key/web files);
    fall back to the two CSV annotator copies."""
    js = sorted(p for p in OUT.glob("paradox_*.json")
                if p.name not in ("paradox_key.json",))
    if js:
        # one annotator is enough; a second file, if present, adds agreement stats
        return js[0], (js[1] if len(js) > 1 else OUT / "_no_second_annotator.json")
    return OUT / "paradox_annotator1.csv", OUT / "paradox_annotator2.csv"


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def perm_p(delta_vec):
    d = np.asarray(delta_vec, float)
    obs = d.mean()
    sg = rng.choice([-1, 1], size=(10000, len(d)))
    return obs, float((np.abs((sg * d).mean(1)) >= abs(obs)).mean())


def main():
    key = json.loads((OUT / "paradox_key.json").read_text())
    f1, f2 = find_annot_files()
    a1 = read_annot(f1) if Path(f1).exists() else {}
    a2 = read_annot(f2) if Path(f2).exists() else {}
    print(f"annotator files: {Path(f1).name} ({len(a1)}), {Path(f2).name} ({len(a2)})")
    if not a1 and not a2:
        print("No annotations found yet. Drop the two exported paradox_<initials>.json "
              "files into the human_eval/ folder.")
        return

    # average the annotators where both present, else use whichever exists
    human = {}
    for uid in set(a1) | set(a2):
        vals = [x[uid] for x in (a1, a2) if uid in x]
        human[uid] = sum(vals) / len(vals)

    # 2. inter-annotator
    both = sorted(set(a1) & set(a2))
    if both:
        r = pearson([a1[u] for u in both], [a2[u] for u in both])
        exact = np.mean([a1[u] == a2[u] for u in both])
        print(f"[inter-annotator] n={len(both)}  Pearson r={r:.3f}  exact-agree={exact:.2f}")

    # 1. paradox under human scoring (paired by query_id)
    by_q = {}
    for uid, h in human.items():
        meta = key[uid]
        by_q.setdefault(meta["query_id"], {})[meta["system"]] = (h, meta["llm_faithfulness"])
    paired = [(v["monolithic"], v["masdr_rag"]) for v in by_q.values()
              if "monolithic" in v and "masdr_rag" in v]
    if paired:
        hm = np.array([p[0][0] for p in paired]); ha = np.array([p[1][0] for p in paired])
        obs, p = perm_p(hm - ha)
        print(f"\n[paradox / HUMAN] n={len(paired)} paired queries")
        print(f"  human faithfulness: Monolithic={hm.mean():.3f}  MASDR-RAG={ha.mean():.3f}  "
              f"Delta={obs:+.3f}  p={p:.3f}")
        verdict = ("SURVIVES human scoring (MASDR < Monolithic)"
                   if obs > 0 and p < 0.05 else
                   "does NOT reach significance under human scoring")
        print(f"  -> paradox {verdict}")

    # 3. human vs LLM judge
    hv = [human[u] for u in human]
    lv = [key[u]["llm_faithfulness"] for u in human]
    print(f"\n[human vs LLM judge] n={len(hv)}  Pearson r={pearson(hv, lv):.3f}")
    print("  (low r => the LLM judge and humans disagree on faithfulness; "
          "high r => LLM judge tracks human judgment)")


if __name__ == "__main__":
    main()
