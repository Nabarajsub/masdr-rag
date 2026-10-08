#!/usr/bin/env python3
"""Blind human-evaluation sheet for the precision-faithfulness paradox.

Reviewer mFpD: the paradox is scored by a single LLM-judge family, and the
cross-family judge gives kappa~0.05 on faithfulness, so "orchestration lowers
faithfulness" is not cleanly separable from "the Qwen judge dislikes
orchestrated answers." This builds a blind sheet so 2+ human annotators can
rate faithfulness independently; we then compute human-vs-LLM agreement and a
paired human faithfulness delta (Monolithic vs MASDR-RAG on WYDOT).

Design:
  * Same queries for both systems (paired), so an annotator can be blind to
    which system produced which answer.
  * Per (query, system) each answer is shown with ITS OWN retrieved sources,
    labeled A/B in randomized order per query.
  * Annotator marks: faithful in {1=fully supported, 0.5=partly, 0=unsupported}
    and an unsupported-claim count. Faithfulness only — NOT correctness — to
    match the RAGAS-style construct under dispute.
  * key.json holds the blind->system map and the hidden LLM-judge scores for
    later agreement analysis. Annotators never see key.json.

Usage:
    python -m emnlp_evaluation.analysis.build_human_eval_paradox --n 50 --seed 7
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

REPO = Path("<DATA_ROOT>")
RES = REPO / "emnlp_evaluation" / "results"
CHUNKS = REPO / "emnlp_evaluation" / "judge_assets" / "wydotv3_chunks.json"
OUT = REPO / "emnlp_evaluation" / "human_eval"

MONO = RES / "wydot_hand_qwen_promptctl_singlecall.judged.jsonl"
MASDR = RES / "wydot_hand_qwen_promptctl_masdr.judged.jsonl"


def load(path, system):
    d = {}
    for line in open(path):
        r = json.loads(line)
        if "error" not in r and r.get("system") == system:
            d[r["query_id"]] = r
    return d


def fmt_sources(chunk_ids, chunk_db, limit=15):
    parts = []
    for i, cid in enumerate(chunk_ids[:limit], 1):
        txt = chunk_db.get(str(cid), "[chunk text unavailable]")
        parts.append(f"[Source {i}] {txt.strip()[:600]}")
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    mono = load(MONO, "monolithic")
    masdr = load(MASDR, "masdr_rag")
    chunk_db = json.loads(CHUNKS.read_text())
    shared = sorted(set(mono) & set(masdr))

    # Sample: half where the LLM judge disagrees most (mono faithful, masdr not)
    # -- the paradox-carrying queries -- and half at random for calibration.
    def faith(r):
        f = r.get("faithfulness")
        return f if f is not None else 0.0
    gap = sorted(shared, key=lambda q: faith(masdr[q]) - faith(mono[q]))
    n_gap = args.n // 2
    picked = gap[:n_gap]
    rest = [q for q in shared if q not in set(picked)]
    picked += rng.sample(rest, min(args.n - n_gap, len(rest)))
    rng.shuffle(picked)

    OUT.mkdir(parents=True, exist_ok=True)
    rows, key = [], {}
    for qid in picked:
        # randomize A/B per query so annotators can't infer system from position
        systems = [("monolithic", mono[qid]), ("masdr_rag", masdr[qid])]
        rng.shuffle(systems)
        for label, (sysname, rec) in zip(["A", "B"], systems):
            uid = f"{qid}_{label}"
            rows.append({
                "uid": uid,
                "query": rec.get("query", ""),
                "sources": fmt_sources(rec.get("chunk_ids", []), chunk_db),
                "answer": rec.get("answer", ""),
                "faithful_1_0.5_0": "",          # annotator fills
                "unsupported_claim_count": "",   # annotator fills
                "notes": "",
            })
            key[uid] = {"query_id": qid, "system": sysname,
                        "llm_faithfulness": faith(rec),
                        "llm_correctness": rec.get("correctness")}

    # web-tool items: blind, one entry per (query, system); sources as a list so
    # the HTML annotator can render them cleanly. No system identity, no scores.
    web = []
    for r in rows:
        meta = key[r["uid"]]
        rec = (mono if meta["system"] == "monolithic" else masdr)[meta["query_id"]]
        srcs = [chunk_db.get(str(c), "[chunk text unavailable]").strip()[:500]
                for c in (rec.get("chunk_ids") or [])[:10]]
        web.append({"uid": r["uid"], "query": r["query"],
                    "answer": r["answer"], "sources": srcs})
    (OUT / "web_items.json").write_text(json.dumps(web, indent=0))

    # annotation sheet (annotators edit this; no system identity, no LLM scores)
    sheet = OUT / "paradox_annotation_sheet.csv"
    with open(sheet, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    # a copy per annotator
    for a in ("annotator1", "annotator2"):
        (OUT / f"paradox_{a}.csv").write_text(sheet.read_text())
    (OUT / "paradox_key.json").write_text(json.dumps(key, indent=2))
    (OUT / "README.md").write_text(
        "# Paradox human eval\n\n"
        f"{len(rows)} answers ({len(picked)} queries x 2 systems), blind A/B.\n\n"
        "**Task (faithfulness only):** read the query and its Sources, then rate the Answer:\n"
        "- `faithful_1_0.5_0`: 1 = every claim supported by the Sources; "
        "0.5 = partly; 0 = key claims unsupported/contradicted.\n"
        "- `unsupported_claim_count`: number of answer claims not backed by any Source.\n"
        "- Judge grounding-in-sources ONLY, not whether the answer is correct.\n\n"
        "Fill `paradox_annotator1.csv` and `paradox_annotator2.csv` independently. "
        "Do NOT open `paradox_key.json`.\n\n"
        "Then run `score_human_eval_paradox.py` to get human faith deltas and "
        "human-vs-LLM / inter-annotator agreement.\n")
    print(f"wrote {sheet} ({len(rows)} rows), key, 2 annotator copies, README -> {OUT}")
    print(f"paradox-gap queries: {n_gap}; calibration: {len(picked)-n_gap}")


if __name__ == "__main__":
    main()
