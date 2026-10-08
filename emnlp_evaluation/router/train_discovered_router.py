#!/usr/bin/env python3
"""Train twin T2 routers for the hand-vs-discovered scope experiment.

Trains the same BGE-M3 linear probe twice, in the runner's agent label space:
  hand : label = document_series of the query's gold document
  disc : label = "scope_<k>", the discovered cluster of the gold document

Both use the cached query embeddings from train_classifier.py (identical
155-query order: test_suite_200 minus CROSS_DOMAIN / VERSION_COMPARISON).
Identical seed/folds, so CV accuracy differences are attributable to the
scope taxonomy alone.

Also materializes the FAISS artifacts for run_generic_corpus:
  data/wydot/derived/wydotv3.meta.parquet   (adds discovered_scope column)
  data/wydot/derived/wydotv3.embeddings.npy (hardlink to wydot embeddings)
  data/wydot/derived/wydotv3.queries.json   (runner query format, all 200)
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

REPO = Path("<DATA_ROOT>")
GP = REPO / "graph_processing"
SUITE = GP / "evaluation" / "test_suite_200.json"
ART = GP / "emnlp_evaluation" / "router" / "artifacts"
DERIVED = REPO / "data" / "wydot" / "derived"
SCOPE_MAP = DERIVED / "wydot_scope_map_k11.parquet"
EXCLUDE = {"CROSS_DOMAIN", "VERSION_COMPARISON"}


# Gold category -> document_series (mirrors production LABEL_TO_AGENT).
CAT_TO_SERIES = {
    "STANDARD_SPECS": "Standard Specs",
    "CONSTRUCTION_MANUAL": "Construction Manual",
    "MATERIALS_TESTING": "Materials Testing",
    "DESIGN_MANUAL": "Design Manual",
    "TRAFFIC_CRASHES": "Traffic & Safety",
    "HIGHWAY_SAFETY": "Traffic & Safety",
    "BRIDGE_PROGRAM": "Bridge Program",
    "STIP": "STIP",
    "ANNUAL_REPORT": "Annual Reports",
    "GENERAL": "General",
}


def gold_maps():
    """Returns (title->cluster map, series->plurality-cluster-by-chunk-mass map)."""
    meta = pd.read_parquet(DERIVED / "wydot.meta.parquet")
    smap = pd.read_parquet(SCOPE_MAP)
    lookup = dict(zip(smap["source"], smap["discovered_scope"]))
    meta = meta.assign(disc=meta["source"].map(lookup))
    # series -> cluster holding the plurality of that series' chunks
    series_map = (meta.groupby(["document_series", "disc"]).size()
                  .groupby(level=0).idxmax().map(lambda t: f"scope_{int(t[1])}")
                  .to_dict())
    # title -> modal cluster of its own doc(s) (32 queries carry a gold title)
    title_map = (meta.groupby("title")["disc"]
                 .agg(lambda s: f"scope_{int(s.mode().iat[0])}").to_dict())
    return title_map, series_map


def gold_labels(row, title_map, series_map):
    """(hand_series, disc_scope) for one suite row."""
    series = CAT_TO_SERIES.get(row["category"], "General")
    title = row.get("relevant_title")
    disc = title_map.get(title) if title else None
    return series, disc or series_map[series]


def cv(name, emb, y, seed=0):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accs, f1s = [], []
    y = np.asarray(y)
    for tr, te in skf.split(np.zeros(len(y)), y):
        clf = LogisticRegression(max_iter=4000, C=1.0, class_weight="balanced")
        clf.fit(emb[tr], y[tr])
        yp = clf.predict(emb[te])
        accs.append(accuracy_score(y[te], yp))
        f1s.append(f1_score(y[te], yp, average="weighted", zero_division=0))
    print(f"[{name}] 5-fold CV acc={np.mean(accs):.3f}±{np.std(accs):.3f} "
          f"F1={np.mean(f1s):.3f}±{np.std(f1s):.3f}  ({len(set(y))} classes, n={len(y)})")
    return {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
            "n": len(y), "classes": sorted(set(y))}


def fit_save(name, emb, y):
    clf = LogisticRegression(max_iter=4000, C=1.0, class_weight="balanced")
    clf.fit(emb, y)
    path = ART / f"{name}.pkl"
    pickle.dump({"model": clf, "labels": list(clf.classes_)}, open(path, "wb"))
    print(f"[{name}] saved {path.name}")


def main():
    rows = json.loads(SUITE.read_text())
    train_rows = [r for r in rows if r["category"] not in EXCLUDE]
    emb = np.load(ART / "wydot_query_bge_m3.npy")
    assert len(train_rows) == emb.shape[0], (len(train_rows), emb.shape)

    title_map, series_map = gold_maps()
    pairs = [gold_labels(r, title_map, series_map) for r in train_rows]
    y_hand = [p[0] for p in pairs]
    y_disc = [p[1] for p in pairs]
    n_title = sum(1 for r in train_rows
                  if r.get("relevant_title") and r["relevant_title"] in title_map)
    print(f"disc gold via own doc title: {n_title}/{len(train_rows)}; "
          f"rest via series plurality map: {series_map}")
    print("hand label dist:", dict(pd.Series(y_hand).value_counts()))
    print("disc label dist:", dict(pd.Series(y_disc).value_counts()))

    results = {
        "hand_series": cv("hand_series", emb, y_hand),
        "disc_k11": cv("disc_k11", emb, y_disc),
    }
    fit_save("T2_bge_logreg_hand_series", emb, y_hand)
    fit_save("T2_bge_logreg_disc_k11", emb, y_disc)
    (ART / "discovered_router_results.json").write_text(json.dumps(results, indent=2))

    # ── FAISS artifacts for run_generic_corpus ──
    meta = pd.read_parquet(DERIVED / "wydot.meta.parquet")
    smap = pd.read_parquet(SCOPE_MAP)
    lookup = dict(zip(smap["source"], smap["discovered_scope"]))
    meta["discovered_scope"] = meta["source"].map(
        lambda s: f"scope_{int(lookup[s])}")
    v3_meta = DERIVED / "wydotv3.meta.parquet"
    meta.to_parquet(v3_meta)
    v3_emb = DERIVED / "wydotv3.embeddings.npy"
    if not v3_emb.exists():
        os.link(DERIVED / "wydot.embeddings.npy", v3_emb)
    queries = []
    for r in rows:
        hand, disc = (gold_labels(r, title_map, series_map)
                      if r["category"] not in EXCLUDE else (None, None))
        queries.append({
            "query_id": r["id"], "query": r["query"],
            "reference_answer": r["reference_answer"],
            "category": r["category"], "question_type": r["query_type"],
            "gold_titles": [r["relevant_title"]] if r.get("relevant_title") else [],
            "gold_hand_scope": hand, "gold_disc_scope": disc,
        })
    (DERIVED / "wydotv3.queries.json").write_text(json.dumps(queries, indent=2))
    print(f"wrote {v3_meta.name}, {v3_emb.name} (hardlink), wydotv3.queries.json "
          f"({len(queries)} queries)")


if __name__ == "__main__":
    main()
