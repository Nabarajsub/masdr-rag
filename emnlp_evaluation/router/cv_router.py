"""5-fold stratified CV for the R0/R1/R2 routers with bootstrap CIs.

Extends router/train_classifier.py: instead of reporting a single
fold-aggregate accuracy, we (a) keep the existing 5-fold split,
(b) bootstrap 1000 resamples of the per-fold predictions to get
95% CIs on accuracy and F1.

Outputs:
    router/artifacts/cv_results.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path("<DATA_ROOT>")
OUT = ROOT / "artifacts" / "cv_results.json"

EXCLUDE = {"CROSS_DOMAIN", "VERSION_COMPARISON"}
N_BOOT = 1000
SEED = 0


def _load_data():
    """Mirrors what train_classifier.py loads: queries + labels + BGE embeddings."""
    qfile = Path("<DATA_ROOT>")
    suite = json.load(open(qfile))
    rows = []
    for q in suite:
        lbl = (q.get("category") or "").upper().strip()
        if not lbl or lbl in EXCLUDE:
            continue
        rows.append({"query": q.get("query", ""), "label": lbl})
    emb = np.load(ROOT / "artifacts" / "wydot_query_bge_m3.npy")
    return rows, emb


def boot_ci(arr, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    arr = np.asarray(arr, dtype=np.float32)
    means = np.array([rng.choice(arr, size=len(arr), replace=True).mean()
                      for _ in range(n_boot)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(arr.mean()), float(lo), float(hi)


def main():
    rows, emb = _load_data()
    print(f"[load] {len(rows)} labeled queries; emb shape {emb.shape}")
    if emb.shape[0] != len(rows):
        # The .npy is keyed on the full 200-query suite; subset to labelled rows.
        full = json.load(open("<DATA_ROOT>"))
        keep_idx = []
        ri = 0
        for i, q in enumerate(full):
            lbl = (q.get("category") or "").upper().strip()
            if lbl and lbl not in EXCLUDE:
                keep_idx.append(i)
                ri += 1
        emb = emb[keep_idx]
        print(f"[align] using {emb.shape[0]} embedding rows")

    queries = [r["query"] for r in rows]
    labels = np.array([r["label"] for r in rows])

    # Routers to CV
    routers = {
        "R1_tfidf_lr": ("tfidf", LogisticRegression(class_weight="balanced", max_iter=4000)),
        "R2_bge_lr":   ("bge",   LogisticRegression(class_weight="balanced", max_iter=4000)),
    }
    results = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for name, (feat, clf) in routers.items():
        accs = []; f1s = []
        for fold, (tr, te) in enumerate(skf.split(np.zeros(len(labels)), labels)):
            if feat == "tfidf":
                vec = TfidfVectorizer(ngram_range=(1, 2))
                Xtr = vec.fit_transform([queries[i] for i in tr])
                Xte = vec.transform([queries[i] for i in te])
            else:
                Xtr = emb[tr]; Xte = emb[te]
            clf.fit(Xtr, labels[tr])
            pred = clf.predict(Xte)
            accs.append(accuracy_score(labels[te], pred))
            f1s.append(f1_score(labels[te], pred, average="weighted"))
        a_m, a_lo, a_hi = boot_ci(accs)
        f_m, f_lo, f_hi = boot_ci(f1s)
        results[name] = {
            "fold_acc":  accs, "fold_f1": f1s,
            "acc_mean": a_m, "acc_ci": [a_lo, a_hi],
            "f1_mean":  f_m, "f1_ci":  [f_lo, f_hi],
        }
        print(f"{name:<18}  acc={a_m:.3f} [{a_lo:.3f},{a_hi:.3f}]  "
              f"F1={f_m:.3f} [{f_lo:.3f},{f_hi:.3f}]")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"[wrote] {OUT}")


if __name__ == "__main__":
    main()
