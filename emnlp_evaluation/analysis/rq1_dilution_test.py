"""RQ1 deep test: does retrieval quality degrade as the WYDOT corpus grows?

Three subcommands:

  build      Write small-corpus subsets (gold documents pinned + random
             distractors) as <out_dir>/wydot{size}_s{seed}.{meta.parquet,
             embeddings.npy,queries.json}, so the generation harness can run
             end-to-end on them. Seed s draws with random.Random(1000+s), the
             same scheme as dilution_scaling_curve.py (s=0 reproduces the
             existing wydot54 subset).

  retrieval  Retrieval-only experiment, no LLM. Two query sets:
               doc   -- the 32 queries carrying gold_titles; relevance = chunk
                        comes from a gold document; gold docs pinned at every size.
               scope -- a seeded random sample of the 200-query suite, restricted
                        to queries with a non-General gold scope; relevance =
                        chunk's document_series equals the query's scope; corpus
                        subsampled stratified by series (category shares fixed,
                        only absolute size varies).
             For every (query, size, seed) it records P@10, Hit@10, MRR@10,
             nDCG@10, the chance rate (share of relevant chunks in the pool),
             and, for doc mode, a version-lenient P@10 that also counts chunks
             that are near-duplicates (cos >= 0.95) of a gold-document chunk.
             Paired statistics compare the small arm with the full corpus.

  e2e        Paired statistics on judged generation logs: full corpus vs each
             small-corpus seed, on identical queries.
"""
from __future__ import annotations

import argparse, ast, json, math, random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

K = 10


# ----------------------------------------------------------------------------- data
def _gold(v):
    if isinstance(v, str):
        try:
            v = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            v = [v]
    return [str(t) for t in v] if v else []


def load(prefix):
    meta = pd.read_parquet(prefix + ".meta.parquet")
    emb = np.load(prefix + ".embeddings.npy", mmap_mode="r")
    queries = json.load(open(prefix + ".queries.json"))
    return meta, emb, queries


def doc_queries(queries):
    return [q for q in queries if _gold(q.get("gold_titles"))]


def draw_doc_subset(titles_sorted, gold_docs, size, seed):
    distractors = sorted(set(titles_sorted) - set(gold_docs))
    rng = random.Random(1000 + seed)
    n_extra = max(0, size - len(gold_docs))
    return set(gold_docs) | set(rng.sample(distractors, min(n_extra, len(distractors))))


def draw_scope_subset(doc_series, size, seed):
    rng = random.Random(1000 + seed)
    bys = {}
    for t, s in sorted(doc_series.items()):
        bys.setdefault(s, []).append(t)
    n_total = len(doc_series)
    keep = set()
    for s, docs in sorted(bys.items()):
        n_s = max(1, round(size * len(docs) / n_total))
        keep |= set(rng.sample(sorted(docs), min(n_s, len(docs))))
    return keep


# ----------------------------------------------------------------------------- build
def cmd_build(a):
    meta, emb, queries = load(a.prefix)
    qs = doc_queries(queries)
    gold_docs = sorted({t for q in qs for t in _gold(q["gold_titles"])})
    titles = meta.title.astype(str).values
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for seed in range(a.seeds):
        keep = draw_doc_subset(sorted(set(titles)), gold_docs, a.size, seed)
        sel = np.isin(titles, list(keep))
        stem = out / f"wydot{a.size}_s{seed}"
        meta[sel].reset_index(drop=True).to_parquet(str(stem) + ".meta.parquet")
        np.save(str(stem) + ".embeddings.npy", np.asarray(emb[np.flatnonzero(sel)]))
        json.dump(qs, open(str(stem) + ".queries.json", "w"), indent=1)
        print(f"[build] {stem.name}: {len(keep)} docs, {sel.sum()} chunks, {len(qs)} queries")


# ----------------------------------------------------------------------------- metrics
def _metrics(rel):
    """rel: boolean array over the ranked top-K."""
    rel = np.asarray(rel, dtype=float)
    hits = np.flatnonzero(rel)
    dcg = float(np.sum(rel / np.log2(np.arange(2, len(rel) + 2))))
    n_rel = int(rel.sum())
    idcg = float(np.sum(1.0 / np.log2(np.arange(2, max(n_rel, 1) + 2)))) if n_rel else 1.0
    return {"p10": rel.sum() / K, "hit10": float(n_rel > 0),
            "mrr10": 1.0 / (hits[0] + 1) if len(hits) else 0.0,
            "ndcg10": dcg / idcg if n_rel else 0.0}


def embed(texts):
    from emnlp_evaluation.embeddings import get_embedder
    V = np.asarray(get_embedder("bge_m3").embed_documents(texts), dtype=np.float32)
    return V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)


def paired(x_small, x_full, n_boot=10000, seed=0):
    x_small, x_full = np.asarray(x_small, float), np.asarray(x_full, float)
    d = x_full - x_small
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    nz = d[d != 0]
    w = stats.wilcoxon(nz) if len(nz) >= 1 else None
    # exact paired sign-flip permutation (Monte Carlo)
    signs = rng.choice([-1, 1], size=(n_boot, len(d)))
    perm = (np.abs((signs * d).mean(1)) >= abs(d.mean()) - 1e-12).mean()
    sd = d.std(ddof=1)
    return {"n": int(len(d)), "small": float(x_small.mean()), "full": float(x_full.mean()),
            "diff": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "wilcoxon_p": float(w.pvalue) if w is not None else 1.0,
            "perm_p": float(perm), "dz": float(d.mean() / sd) if sd > 0 else float("nan"),
            "n_worse": int((d < 0).sum()), "n_better": int((d > 0).sum()), "n_tied": int((d == 0).sum())}


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i])); adj[i] = running
    return adj


def trend(df, metric):
    """Per-query Spearman(log n_docs, metric) across all (size, seed) cells, then a
    one-sample Wilcoxon on the per-query rhos; plus a query-cluster bootstrap of
    the pooled OLS slope per log10(docs)."""
    rhos = []
    for _, g in df.groupby("qid"):
        if g[metric].nunique() > 1:
            rhos.append(stats.spearmanr(np.log(g.n_docs), g[metric]).statistic)
    rhos = np.array(rhos)
    w = stats.wilcoxon(rhos) if len(rhos) else None
    qids = df.qid.unique(); rng = np.random.default_rng(1)
    by = {q: g for q, g in df.groupby("qid")}
    def slope(sub):
        return np.polyfit(np.log10(sub.n_docs), sub[metric], 1)[0]
    s0 = slope(df)
    bs = [slope(pd.concat([by[q] for q in rng.choice(qids, len(qids))])) for _ in range(2000)]
    return {"n_queries_varying": int(len(rhos)), "median_rho": float(np.median(rhos)) if len(rhos) else None,
            "frac_negative": float((rhos < 0).mean()) if len(rhos) else None,
            "wilcoxon_p": float(w.pvalue) if w is not None else None,
            "slope_per_log10_docs": float(s0),
            "slope_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


# ----------------------------------------------------------------------------- retrieval
def cmd_retrieval(a):
    meta, emb, queries = load(a.prefix)
    titles = meta.title.astype(str).values
    series = meta.document_series.astype(str).values
    doc_series = dict(zip(titles, series))
    all_docs = sorted(set(titles))
    N = len(all_docs)
    E = np.array(emb, dtype=np.float32)
    E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)

    # --- query sets
    dq = doc_queries(queries)
    rng = random.Random(a.sample_seed)
    sample = rng.sample(queries, min(a.sample_n, len(queries)))
    sq = [q for q in sample if q.get("gold_hand_scope") and str(q["gold_hand_scope"]) != "General"]
    print(f"[data] {len(meta)} chunks / {N} docs; doc-mode {len(dq)} q; "
          f"scope-mode {len(sq)} of a random {len(sample)} (seed {a.sample_seed})")

    Qd = embed([q["query"] for q in dq]); Qs = embed([q["query"] for q in sq])
    Sd = Qd @ E.T; Ss = Qs @ E.T
    gold_docs = sorted({t for q in dq for t in _gold(q["gold_titles"])})

    # near-duplicate map for version-lenient scoring: for each doc query, the set of
    # chunks with cos >= 0.95 to any chunk of its gold documents.
    dup_sets = []
    for q in dq:
        gidx = np.flatnonzero(np.isin(titles, _gold(q["gold_titles"])))
        cand = np.argsort(-Sd[len(dup_sets)])[:5000]          # only top-5000 can reach top-10
        sim = E[cand] @ E[gidx].T
        dup_sets.append(set(cand[sim.max(1) >= 0.95].tolist()))

    sizes = [int(x) for x in a.sizes.split(",")]
    rows = []
    for mode, qs, S in (("doc", dq, Sd), ("scope", sq, Ss)):
        for size in sizes:
            full = size >= N
            for seed in range(1 if full else a.seeds):
                if full:
                    keep = set(all_docs)
                elif mode == "doc":
                    keep = draw_doc_subset(all_docs, gold_docs, size, seed)
                else:
                    keep = draw_scope_subset(doc_series, size, seed)
                idx = np.flatnonzero(np.isin(titles, list(keep)))
                for i, q in enumerate(qs):
                    sims = S[i, idx]
                    top = idx[np.argsort(-sims)[:K]]
                    if mode == "doc":
                        gset = set(_gold(q["gold_titles"]))
                        rel = np.array([titles[j] in gset for j in top])
                        chance = float(np.isin(titles[idx], list(gset)).mean())
                        len_rel = np.array([(titles[j] in gset) or (j in dup_sets[i]) for j in top])
                        gser = {doc_series.get(t) for t in gset}
                        same_series_nongold = float(np.mean([(titles[j] not in gset) and (series[j] in gser) for j in top]))
                        other_series = float(np.mean([series[j] not in gser for j in top]))
                    else:
                        rel = np.array([series[j] == str(q["gold_hand_scope"]) for j in top])
                        chance = float((series[idx] == str(q["gold_hand_scope"])).mean())
                        len_rel = rel; same_series_nongold = float("nan"); other_series = float(1 - rel.mean())
                    m = _metrics(rel)
                    rows.append({"mode": mode, "qid": q["query_id"], "size": size, "seed": seed,
                                 "n_docs": len(keep), "n_chunks": int(len(idx)), **m,
                                 "p10_lenient": float(len_rel.mean()), "chance": chance,
                                 "lift": m["p10"] / chance if chance > 0 else float("nan"),
                                 "nongold_same_series": same_series_nongold,
                                 "other_series": other_series,
                                 "qtype": q.get("question_type")})
            print(f"  {mode} size={size} done")
    df = pd.DataFrame(rows)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(out).replace(".json", ".rows.csv"), index=False)

    report = {"config": vars(a), "n_chunks": int(len(meta)), "n_docs": N}
    metrics = ["p10", "hit10", "mrr10", "ndcg10", "p10_lenient", "lift"]
    for mode in ("doc", "scope"):
        d = df[df["mode"] == mode]
        curve = d.groupby("size")[metrics + ["chance", "n_chunks", "nongold_same_series", "other_series"]].mean()
        seed_sd = d.groupby(["size", "seed"])[metrics].mean().groupby("size").std()
        report[f"{mode}_curve"] = {"mean": json.loads(curve.to_json()), "between_seed_sd": json.loads(seed_sd.to_json())}
        small = d[d["size"] == a.small].groupby("qid")[metrics].mean()
        full = d[d["n_docs"] == N].set_index("qid")[metrics]
        common = small.index.intersection(full.index)
        tests = {m: paired(small.loc[common, m], full.loc[common, m]) for m in metrics}
        adj = holm(np.array([tests[m]["wilcoxon_p"] for m in metrics]))
        for m, p in zip(metrics, adj):
            tests[m]["wilcoxon_p_holm"] = float(p)
        report[f"{mode}_small_vs_full"] = tests
        # per-seed replication: does every individual distractor draw show the drop?
        per_seed = []
        for seed, g in d[d["size"] == a.small].groupby("seed"):
            g = g.set_index("qid").loc[common]
            per_seed.append({"seed": int(seed), **{m: paired(g[m], full.loc[common, m], n_boot=2000)["wilcoxon_p"] for m in ("p10", "hit10")},
                             "p10_small": float(g.p10.mean()), "hit10_small": float(g.hit10.mean())})
        report[f"{mode}_per_seed"] = per_seed
        report[f"{mode}_trend"] = {m: trend(d, m) for m in ("p10", "hit10", "ndcg10", "lift")}
        if mode == "doc":
            report["doc_by_qtype"] = json.loads(d[d["n_docs"] == N].groupby("qtype")[["p10", "hit10"]].mean().to_json())
    Path(a.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k.endswith("small_vs_full")}, indent=1))


# ----------------------------------------------------------------------------- e2e
def _read(path):
    return {r["query_id"]: r for r in (json.loads(l) for l in open(path)) if "error" not in r}


def cmd_e2e(a):
    full = _read(a.full)
    seeds = [_read(p) for p in a.small]
    common = sorted(set(full).intersection(*[set(s) for s in seeds]))
    out = {"n_queries": len(common), "full_file": a.full, "small_files": a.small}

    def gold_hit(r):
        g = set(_gold(r.get("gold_titles")))
        return float(any(str(t) in g for t in (r.get("chunk_titles") or [])))

    def gold_p(r):
        g = set(_gold(r.get("gold_titles"))); t = [str(x) for x in (r.get("chunk_titles") or [])]
        return sum(x in g for x in t) / max(1, len(t))

    for m, f in (("correctness", lambda r: float(r["correctness"])),
                 ("faithfulness", lambda r: float(r["faithfulness"])),
                 ("gold_doc_hit@ctx", gold_hit), ("gold_doc_precision@ctx", gold_p)):
        fv = np.array([f(full[q]) for q in common])
        sv = np.array([[f(s[q]) for q in common] for s in seeds])
        res = paired(sv.mean(0), fv)
        res["per_seed_small_means"] = [float(x) for x in sv.mean(1)]
        if m == "correctness":
            mc = []
            for row in sv:
                b = int(((row == 1) & (fv == 0)).sum()); c = int(((row == 0) & (fv == 1)).sum())
                p = stats.binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
                mc.append({"small_right_full_wrong": b, "small_wrong_full_right": c, "mcnemar_exact_p": float(p)})
            res["mcnemar_per_seed"] = mc
            # minimum detectable drop (80% power, alpha .05, McNemar) given observed discordance
            res["note"] = "see mde_* for power"
        out[m] = res
    # does losing the gold document from context predict a wrong answer? (full arm + all seeds pooled)
    pairs = [(gold_hit(r), float(r["correctness"])) for s in [full] + seeds for r in (s[q] for q in common)]
    ph = np.array(pairs)
    tab = [[int(((ph[:, 0] == h) & (ph[:, 1] == c)).sum()) for c in (0, 1)] for h in (0, 1)]
    out["correct_given_gold_in_ctx"] = {"table_[hit][correct]": tab,
                                         "fisher_p": float(stats.fisher_exact(tab).pvalue),
                                         "acc_hit": tab[1][1] / max(1, sum(tab[1])),
                                         "acc_miss": tab[0][1] / max(1, sum(tab[0]))}
    # power: small arm at its observed accuracy; the full arm is a correlated copy
    # (latent-uniform coupling, rho chosen so the no-effect discordance matches the
    # ~10% observed between draws) shifted down by `drop`. Exact McNemar at .05.
    p_small = out["correctness"]["small"]; n = len(common); rng = np.random.default_rng(0)
    rho, mde = 0.62, None
    for drop in np.arange(0.02, p_small, 0.02):
        sig = 0
        for _ in range(1000):
            u = rng.random(n); v = np.where(rng.random(n) < rho, u, rng.random(n))
            sm, fu = u < p_small, v < p_small - drop
            b = int((sm & ~fu).sum()); c = int((~sm & fu).sum())
            if b + c and stats.binomtest(min(b, c), b + c, 0.5).pvalue < 0.05:
                sig += 1
        if sig / 1000 >= 0.8:
            mde = float(drop); break
    out["mde_correctness_drop_80pct_power"] = mde
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--prefix", required=True); b.add_argument("--out-dir", required=True)
    b.add_argument("--size", type=int, default=54); b.add_argument("--seeds", type=int, default=5)
    r = sub.add_parser("retrieval"); r.add_argument("--prefix", required=True); r.add_argument("--out", required=True)
    r.add_argument("--sizes", default="16,54,100,200,400,700,1083"); r.add_argument("--seeds", type=int, default=20)
    r.add_argument("--small", type=int, default=54)
    r.add_argument("--sample-n", type=int, default=100); r.add_argument("--sample-seed", type=int, default=2026)
    e = sub.add_parser("e2e"); e.add_argument("--full", required=True); e.add_argument("--small", nargs="+", required=True)
    e.add_argument("--out", required=True)
    a = ap.parse_args()
    {"build": cmd_build, "retrieval": cmd_retrieval, "e2e": cmd_e2e}[a.cmd](a)


if __name__ == "__main__":
    main()
