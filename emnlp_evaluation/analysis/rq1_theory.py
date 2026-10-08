"""RQ1 theory: why does gold-document P@10 fall as the corpus grows?

Order-statistics model. For query q let s_g be its gold-chunk similarities and D_q(s) the
number of non-gold chunks in the FULL corpus with similarity > s. If a fraction r of the
non-gold material is present, the number of competitors above s is ~Binomial(D_q(s), r),
and gold chunk g is in the top-k iff  #(gold above s_g) + #(kept competitors above s_g) < k.
Nothing about the embedding changes with r -- only how many competitors reach the tail.

Tests:
  1. PREDICT the measured doc-sampled curve (RQ1 E21) from the full-corpus similarity lists
     alone, by chunk-level thinning at the matching rate r (Monte Carlo, 200 draws).
  2. Discrimination is size-invariant: AUC(gold sims vs non-gold sims) per query, and the
     'tail count' D_q(median gold sim).
  3. Extreme values: E[max competitor sim] vs N, fit a + b*sqrt(2 ln N) (Gaussian-tail law).
  4. Tail composition above each query's 10th-best gold chunk: editions / same-category
     siblings / other categories, with AUC(gold vs that population) -- editions ~ 0.5 means
     the embedding cannot separate them even in principle.
  5. Geometry: anisotropy (mean cosine of random chunk pairs), query-similarity spread,
     hubness (skewness of 10-occurrence counts).
"""
from __future__ import annotations
import argparse, ast, json, re
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

K = 10


def _gold(v):
    if isinstance(v, str):
        v = ast.literal_eval(v)
    return [str(t) for t in v] if v else []


def _stem(t):
    t = re.sub(r"\b(19|20)\d{2}\b|\bfy\s*\d+\b|\brev(ision)?\b|\bupdated\b|\(.*?\)|effective.*|[^a-z ]", " ", t.lower())
    return " ".join(w for w in t.split() if w not in {"wydot", "wyoming", "the", "for", "and", "of"})


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(stats.mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--curve", required=True, help="rq1_retrieval.rows.csv from E21")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    meta = pd.read_parquet(a.prefix + ".meta.parquet")
    E = np.load(a.prefix + ".embeddings.npy").astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    titles = meta.title.astype(str).values; series = meta.document_series.astype(str).values
    stems = np.array([_stem(t) for t in titles])
    Q = [q for q in json.load(open(a.prefix + ".queries.json")) if _gold(q.get("gold_titles"))]
    from emnlp_evaluation.embeddings import get_embedder
    V = np.asarray(get_embedder("bge_m3").embed_documents([q["query"] for q in Q]), np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    S = V @ E.T
    rep = {"n_queries": len(Q), "n_chunks": int(len(E))}

    # --- measured curve (doc-sampled, E21)
    rows = pd.read_csv(a.curve); rows = rows[rows["mode"] == "doc"]
    curve = rows.groupby("size").agg(p10=("p10", "mean"), n_chunks=("n_chunks", "mean")).reset_index()

    gold_sets = [set(_gold(q["gold_titles"])) for q in Q]
    gmask = [np.isin(titles, list(g)) for g in gold_sets]
    n_gold_total = int(np.unique(np.concatenate([np.flatnonzero(m) for m in gmask])).size)
    rng = np.random.default_rng(0)

    # --- 1. predict the curve by chunk thinning
    pred = []
    for _, row in curve.iterrows():
        n_all_gold_chunks = sum(int(m.sum()) for m in gmask) / len(Q)  # not used for r
        r = max(0.0, (row.n_chunks - n_gold_total) / (len(E) - n_gold_total))
        pq = []
        for i in range(len(Q)):
            s = S[i]; gm = gmask[i]
            order = np.argsort(-s)[:5000]                    # the tail that can matter
            is_g = gm[order]
            hits = []
            for _ in range(200):
                keep = is_g | (rng.random(len(order)) < r)
                hits.append(is_g[keep][:K].mean())
            pq.append(np.mean(hits))
        pred.append({"size": int(row["size"]), "r": round(r, 4), "measured_p10": round(row.p10, 3),
                     "predicted_p10": round(float(np.mean(pq)), 3)})
    rep["curve_prediction"] = pred
    m_ = np.array([p["measured_p10"] for p in pred]); p_ = np.array([p["predicted_p10"] for p in pred])
    rep["curve_prediction_summary"] = {"mean_abs_error": float(np.abs(m_ - p_).mean()),
                                       "pearson_r": float(stats.pearsonr(m_, p_).statistic)}

    # --- 2. discrimination vs tail count
    per_q = []
    for i in range(len(Q)):
        s = S[i]; gm = gmask[i]
        gs, ns = s[gm], s[~gm]
        per_q.append({"qid": Q[i]["query_id"], "auc_gold_vs_all": auc(gs, ns),
                      "gold_best": float(gs.max()), "gold_median": float(np.median(gs)),
                      "n_nongold_above_gold_best": int((ns > gs.max()).sum()),
                      "n_nongold_above_gold_10th": int((ns > np.sort(gs)[-min(K, len(gs))]).sum())})
    pq = pd.DataFrame(per_q)
    rep["discrimination"] = {"auc_median": float(pq.auc_gold_vs_all.median()),
                             "auc_iqr": [float(pq.auc_gold_vs_all.quantile(.25)), float(pq.auc_gold_vs_all.quantile(.75))],
                             "median_nongold_above_gold_best_full": float(pq.n_nongold_above_gold_best.median()),
                             "note": "AUC is a property of the two similarity distributions; random subsampling leaves it unchanged in expectation, "
                                     "while the COUNT above the gold scales ~linearly with r."}

    # --- 3. extreme values of the competitor maximum vs N
    Ns = np.unique(np.logspace(3, np.log10(len(E) - 1000), 12).astype(int))
    ev = []
    for n in Ns:
        mx = []
        for i in range(len(Q)):
            ns_ = S[i][~gmask[i]]
            mx.append(np.mean([ns_[rng.choice(len(ns_), min(n, len(ns_)), replace=False)].max() for _ in range(20)]))
        ev.append((int(n), float(np.mean(mx))))
    x = np.sqrt(2 * np.log([n for n, _ in ev])); y = np.array([m for _, m in ev])
    b, a0 = np.polyfit(x, y, 1)
    rep["extreme_value"] = {"points": ev, "fit": {"a": float(a0), "b": float(b),
                            "r2": float(np.corrcoef(x, y)[0, 1] ** 2)},
                            "gold_best_median": float(pq.gold_best.median())}

    # --- 4. tail composition + separability by population
    comp = {"edition": [], "sibling": [], "other": []}; aucs = {"edition": [], "sibling": [], "other": []}
    for i in range(len(Q)):
        s = S[i]; gm = gmask[i]
        g = gold_sets[i]; gst = {_stem(t) for t in g}
        gser = {series[gm][0]} if gm.any() else set()
        gidx = np.flatnonzero(gm)
        top = np.argsort(-s)[:3000]
        near = np.zeros(len(E), bool)
        near[top] = (E[top] @ E[gidx].T).max(1) >= 0.95
        pop_ed = (~gm) & (np.isin(stems, list(gst)) | near)
        pop_sib = (~gm) & ~pop_ed & np.isin(series, list(gser))
        pop_oth = (~gm) & ~pop_ed & ~pop_sib
        thr = np.sort(s[gm])[-min(K, gm.sum())]
        above = s > thr
        tot = max(1, int((above & ~gm).sum()))
        for name, pop in (("edition", pop_ed), ("sibling", pop_sib), ("other", pop_oth)):
            comp[name].append(int((above & pop).sum()) / tot)
            aucs[name].append(auc(s[gm], s[pop]))
    rep["tail_composition_above_10th_gold"] = {k: float(np.mean(v)) for k, v in comp.items()}
    rep["auc_gold_vs_population"] = {k: float(np.nanmedian(v)) for k, v in aucs.items()}

    # --- 5. geometry
    i1 = rng.integers(0, len(E), 200000); i2 = rng.integers(0, len(E), 200000)
    pair_cos = np.einsum("ij,ij->i", E[i1], E[i2])
    samp = rng.choice(len(E), 3000, replace=False)
    occ = np.zeros(len(E), np.int32)
    for lo in range(0, len(samp), 250):
        blk = E[samp[lo:lo + 250]] @ E.T
        blk[np.arange(blk.shape[0]), samp[lo:lo + 250]] = -1
        nn = np.argpartition(-blk, K, axis=1)[:, :K]
        np.add.at(occ, nn.ravel(), 1)
    rep["geometry"] = {"mean_pair_cosine": float(pair_cos.mean()), "sd_pair_cosine": float(pair_cos.std()),
                       "query_sim_mean": float(S.mean()), "query_sim_sd": float(S.std()),
                       "hubness_skew_N10": float(stats.skew(occ)),
                       "share_of_NN_slots_taken_by_top1pct_chunks": float(np.sort(occ)[::-1][:len(E) // 100].sum() / occ.sum())}
    Path(a.out).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
