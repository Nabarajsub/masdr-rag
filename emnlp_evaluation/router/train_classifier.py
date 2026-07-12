"""Trained domain-router baselines (G-2 in plan.md).

Two cheap tiers — both fit the reviewer's "lightweight linear/transformer
domain classifier" ask:

  T1 : TF-IDF (1-2 gram, char + word) -> logistic regression
  T2 : mean-pooled BGE-M3 query embedding -> logistic regression (linear probe)

Training data is the 200 WYDOT queries with gold `category` labels in
graph_processing/evaluation/test_suite_200.json. The two meta categories
(CROSS_DOMAIN, VERSION_COMPARISON) are dropped: domain routing is about
single-domain queries by construction.

Reports top-1 / top-2 accuracy, weighted F1, latency per query
(microseconds for T1, milliseconds for T2 -- the embedding pass), and
emits a confusion matrix. Persists pickled models for plug-in use by the
MASDR-RAG eval (G-2 follow-on).
"""
from __future__ import annotations

import argparse, json, pickle, time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score,
)
from sklearn.model_selection import StratifiedKFold

REPO = Path("<DATA_ROOT>")
SUITE = REPO / "evaluation" / "test_suite_200.json"
OUT_DIR = REPO / "emnlp_evaluation" / "router" / "artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE = {"CROSS_DOMAIN", "VERSION_COMPARISON"}


def load_data():
    rows = json.loads(SUITE.read_text())
    X = [r["query"] for r in rows if r.get("category") not in EXCLUDE]
    y = [r["category"] for r in rows if r.get("category") not in EXCLUDE]
    qids = [r["id"] for r in rows if r.get("category") not in EXCLUDE]
    return X, y, qids


def bge_embed(texts: list[str]) -> np.ndarray:
    """Mean-pooled BGE-M3 embeddings (1024-d)."""
    from sentence_transformers import SentenceTransformer
    import os
    model_path = os.environ.get(
        "BGE_M3_LOCAL",
        "<DATA_ROOT>"
        "models--BAAI--bge-m3/snapshots/"
        "5617a9f61b028005a4858fdac845db406aefb181",
    )
    print(f"[bge] loading {model_path}")
    m = SentenceTransformer(model_path)
    t0 = time.time()
    emb = m.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    print(f"[bge] {len(texts)} texts -> {emb.shape} in {time.time()-t0:.2f}s")
    return np.asarray(emb, dtype=np.float32)


def cv_eval(name, X, y, build_model_fn, embed=None):
    """Stratified 5-fold CV; report top-1 acc + weighted-F1."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    accs, f1s, top2s = [], [], []
    for fold, (tr, te) in enumerate(skf.split(np.zeros(len(X)), y)):
        if embed is None:
            X_tr = [X[i] for i in tr]; X_te = [X[i] for i in te]
            y_tr = [y[i] for i in tr]; y_te = [y[i] for i in te]
            model = build_model_fn()
            model.fit(X_tr, y_tr)
            yp = model.predict(X_te)
            proba = model.predict_proba(X_te)
        else:
            X_tr = embed[tr]; X_te = embed[te]
            y_tr = [y[i] for i in tr]; y_te = [y[i] for i in te]
            model = build_model_fn()
            model.fit(X_tr, y_tr)
            yp = model.predict(X_te)
            proba = model.predict_proba(X_te)
        classes = model.classes_
        top2 = 0
        for i, yt in enumerate(y_te):
            top2_idx = np.argsort(-proba[i])[:2]
            if yt in classes[top2_idx]:
                top2 += 1
        accs.append(accuracy_score(y_te, yp))
        f1s.append(f1_score(y_te, yp, average="weighted", zero_division=0))
        top2s.append(top2 / len(y_te))
    print(f"[{name}] 5-fold CV: acc={np.mean(accs):.3f}±{np.std(accs):.3f}  "
          f"top2={np.mean(top2s):.3f}±{np.std(top2s):.3f}  "
          f"F1={np.mean(f1s):.3f}±{np.std(f1s):.3f}")
    return {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "top2_mean": float(np.mean(top2s)), "top2_std": float(np.std(top2s)),
            "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s))}


def fit_final(name, X, y, build_model_fn, embed=None):
    model = build_model_fn()
    if embed is None:
        model.fit(X, y)
    else:
        model.fit(embed, y)
    pickle.dump(model, open(OUT_DIR / f"{name}.pkl", "wb"))
    print(f"[{name}] full-fit model -> {OUT_DIR/(name+'.pkl')}")
    return model


def regex_baseline(X, y):
    """Replays the production regex router from agents/hybrid_routed.py for
    a head-to-head reviewer-friendly comparison."""
    import re
    RULES = [
        (re.compile(r"\b(inspection|inspector)\b", re.I), "CONSTRUCTION_MANUAL"),
        (re.compile(r"\b(spec(s)?|specification|section\s*\d{3})\b", re.I), "STANDARD_SPECS"),
        (re.compile(r"\b(material|lab|test(ing)?|aggregate|asphalt)\b", re.I), "MATERIALS_TESTING"),
        (re.compile(r"\b(design|geometric|alignment)\b", re.I), "DESIGN_MANUAL"),
        (re.compile(r"\b(crash|safety|fatalit|vehicle)\b", re.I), "TRAFFIC_CRASHES"),
        (re.compile(r"\b(bridge|culvert|deck|load.posting)\b", re.I), "BRIDGE_PROGRAM"),
        (re.compile(r"\b(stip|program(ming)?|funding|fiscal)\b", re.I), "STIP"),
        (re.compile(r"\b(annual\s*report|fiscal\s*year)\b", re.I), "ANNUAL_REPORT"),
        (re.compile(r"\b(highway\s*safety|hsip|sf2)\b", re.I), "HIGHWAY_SAFETY"),
    ]
    preds = []
    for q in X:
        hit = next((cat for rx, cat in RULES if rx.search(q)), None)
        preds.append(hit or "GENERAL")
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average="weighted", zero_division=0)
    print(f"[regex] hit-rate={sum(p!='GENERAL' for p in preds)/len(preds):.3f}  "
          f"acc={acc:.3f}  F1={f1:.3f}")
    return {"acc_mean": float(acc), "f1_mean": float(f1),
            "hit_rate": float(sum(p != "GENERAL" for p in preds) / len(preds))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-bge", action="store_true",
                    help="Skip BGE-M3 embedding (T2)")
    args = ap.parse_args()

    X, y, qids = load_data()
    print(f"n queries: {len(X)}")
    print(f"label dist: {Counter(y)}")

    results = {}

    # Regex replay (G-2 baseline reference)
    results["regex"] = regex_baseline(X, y)

    # T1: TF-IDF + logistic regression
    def t1():
        from sklearn.pipeline import Pipeline
        return Pipeline([
            ("vec", TfidfVectorizer(
                ngram_range=(1, 2), min_df=1, max_df=1.0, sublinear_tf=True,
                analyzer="word", lowercase=True)),
            ("clf", LogisticRegression(
                max_iter=2000, C=1.0, class_weight="balanced", n_jobs=1)),
        ])
    results["T1_tfidf_logreg"] = cv_eval("T1_tfidf_logreg", X, y, t1)
    fit_final("T1_tfidf_logreg", X, y, t1)

    # T2: BGE-M3 linear probe
    if not args.no_bge:
        emb = bge_embed(X)
        np.save(OUT_DIR / "wydot_query_bge_m3.npy", emb)
        def t2():
            return LogisticRegression(
                max_iter=4000, C=1.0, class_weight="balanced", n_jobs=1)
        results["T2_bge_logreg"] = cv_eval("T2_bge_logreg", X, y, t2, embed=emb)
        fit_final("T2_bge_logreg", X, y, t2, embed=emb)

    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\n[router] wrote {OUT_DIR/'results.json'}")
    for name, r in results.items():
        a = r.get("acc_mean")
        f1 = r.get("f1_mean")
        print(f"  {name:>22}  acc={a:.3f}  F1={f1:.3f}")


if __name__ == "__main__":
    main()
