"""RQ2: does scoping turn retrieval precision into answers, and is it safe?

Pairs every scoped arm with monolithic query-by-query on the E7 shared-prompt logs
(plus any extra arm files, e.g. E22 oracle / soft-sensitivity), and reports:
  - Δcorrectness / Δfaithfulness with paired bootstrap 95% CI, exact McNemar,
    Holm across the whole family of correctness comparisons
  - non-inferiority verdict at margins -0.03 (primary) and -0.05
  - retrieval of the 15-chunk context: scope precision (share of chunks from the
    query's gold scope) and gold-evidence hit (gold chunk / gold document)

Usage:
  python -m emnlp_evaluation.analysis.rq2_scoping_test --out results/rq2/rq2_partA.json \
      [--extra composite:results/rq2/composite_oracle.judged.jsonl ...]
"""
from __future__ import annotations

import argparse, ast, json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

RES = Path(__file__).resolve().parents[1] / "results"
_ROOT = Path(__file__).resolve().parents[2]
# released layout first (data/queries/), then the original cluster layout
WYDOT_Q = next((p for p in (_ROOT / "data/queries/wydotv3_queries.json",
                            _ROOT.parent / "data/wydot/derived/wydotv3.queries.json")
                if p.exists()), _ROOT / "data/queries/wydotv3_queries.json")

BASE = {
    "wydot_hand": ["wydot_hand_qwen_promptctl_singlecall.judged.jsonl"],
    "composite": ["composite_qwen_promptctl_singlecall.judged.jsonl"],
    "multihop": ["multihop_qwen_promptctl_singlecall.judged.jsonl"],
    "mmlu_pro": ["mmlu_pro_qwen_promptctl_singlecall.judged.jsonl"],
    "financebench": ["financebench_qwen_promptctl_singlecall.judged.jsonl"],
    "crag": ["crag_qwen_promptctl_singlecall.judged.jsonl", "crag_qwen_promptctl_soft.judged.jsonl"],
}
MARGINS = (-0.03, -0.05)


def _list(v):
    if isinstance(v, str):
        try:
            v = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            v = [v]
    return [str(x) for x in v] if v else []


def load(corpus, files, tag=None):
    wydot_scope = {}
    if corpus == "wydot_hand":
        wydot_scope = {q["query_id"]: q.get("gold_hand_scope") for q in json.load(open(WYDOT_Q))}
    arms = defaultdict(dict)
    for f in files:
        for line in open(f):
            r = json.loads(line)
            if "error" in r or r.get("correctness") is None:
                continue
            name = tag or r["system"]
            if corpus == "wydot_hand":
                r["_scope"] = wydot_scope.get(r["query_id"])
            elif corpus in ("composite", "mmlu_pro", "financebench"):
                r["_scope"] = r.get("category")
            else:
                r["_scope"] = None
            arms[name][r["query_id"]] = r
    return arms


def retrieval(r, corpus):
    src = [str(s) for s in (r.get("chunk_sources") or [])]
    out = {}
    sc = r.get("_scope")
    if sc and sc != "General" and src:
        out["scope_prec"] = sum(s == str(sc) for s in src) / len(src)
    gcid = r.get("gold_chunk_id")
    if gcid:
        ids = [str(x) for x in (r.get("chunk_ids") or [])]
        out["gold_chunk_hit"] = float(str(gcid) in ids)
        gdoc = "::".join(str(gcid).split("::")[:2])
        out["gold_doc_hit"] = float(any(x.startswith(gdoc + "::") for x in ids))
    gt = _list(r.get("gold_titles"))
    if gt:
        titles = [str(t) for t in (r.get("chunk_titles") or [])]
        out["gold_title_recall"] = sum(t in titles for t in gt) / len(gt)
        out["gold_title_prec"] = sum(t in set(gt) for t in titles) / max(1, len(titles))
    return out


def paired(a, b, rng):
    d = np.asarray(a, float) - np.asarray(b, float)
    boots = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    res = {"n": int(len(d)), "arm": float(np.mean(a)), "mono": float(np.mean(b)),
           "diff": float(d.mean()), "ci95": [float(lo), float(hi)]}
    if set(np.unique(np.concatenate([a, b]))) <= {0.0, 1.0}:
        bb = int(((np.asarray(a) == 1) & (np.asarray(b) == 0)).sum())
        cc = int(((np.asarray(a) == 0) & (np.asarray(b) == 1)).sum())
        res.update(arm_only=bb, mono_only=cc,
                   p=float(stats.binomtest(min(bb, cc), bb + cc, 0.5).pvalue) if bb + cc else 1.0,
                   test="exact McNemar")
    else:
        nz = d[d != 0]
        res.update(p=float(stats.wilcoxon(nz).pvalue) if len(nz) else 1.0, test="Wilcoxon")
    return res


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * ps[i])); adj[i] = run
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra", nargs="*", default=[],
                    help="corpus:path[:tag] -- extra judged arm files paired against monolithic")
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    data = {c: load(c, [RES / f for f in fs]) for c, fs in BASE.items()}
    for spec in a.extra:
        parts = spec.split(":")
        corpus, path = parts[0], parts[1]
        tag = parts[2] if len(parts) > 2 else None
        for name, rows in load(corpus, [path], tag=tag).items():
            data[corpus][name] = rows

    report = {"comparisons": [], "retrieval": {}}
    for corpus, arms in data.items():
        mono = arms.get("monolithic", {})
        report["retrieval"][corpus] = {}
        for name, rows in arms.items():
            vals = defaultdict(list)
            for r in rows.values():
                for k, v in retrieval(r, corpus).items():
                    vals[k].append(v)
            report["retrieval"][corpus][name] = {k: round(float(np.mean(v)), 4) for k, v in vals.items()} | {"n": len(rows)}
            if name == "monolithic":
                continue
            common = sorted(set(rows) & set(mono))
            for metric in ("correctness", "faithfulness"):
                x = np.array([float(rows[q][metric]) for q in common])
                y = np.array([float(mono[q][metric]) for q in common])
                res = paired(x, y, rng)
                res.update(corpus=corpus, system=name, metric=metric)
                for m in MARGINS:
                    res[f"noninferior_{m}"] = bool(res["ci95"][0] > m)
                report["comparisons"].append(res)

    for metric in ("correctness", "faithfulness"):
        fam = [c for c in report["comparisons"] if c["metric"] == metric]
        adj = holm(np.array([c["p"] for c in fam]))
        for c, p in zip(fam, adj):
            c["p_holm"] = float(p)
        report[f"family_size_{metric}"] = len(fam)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2))
    print(f"{'corpus':<13}{'system':<18}{'n':>5}{'arm':>7}{'mono':>7}{'Δ':>8}  {'95% CI':<17}{'p':>8}{'p_holm':>8}  NI.03 NI.05")
    for c in report["comparisons"]:
        if c["metric"] != "correctness":
            continue
        print(f"{c['corpus']:<13}{c['system']:<18}{c['n']:>5}{c['arm']:>7.3f}{c['mono']:>7.3f}{c['diff']:>+8.3f}  "
              f"[{c['ci95'][0]:+.3f},{c['ci95'][1]:+.3f}]{c['p']:>8.3g}{c['p_holm']:>8.3g}  "
              f"{'yes' if c['noninferior_-0.03'] else 'no':>5} {'yes' if c['noninferior_-0.05'] else 'no':>5}")
    print("\nfaithfulness:")
    for c in report["comparisons"]:
        if c["metric"] == "faithfulness":
            print(f"{c['corpus']:<13}{c['system']:<18}{c['arm']:>7.3f}{c['mono']:>7.3f}{c['diff']:>+8.3f}  "
                  f"[{c['ci95'][0]:+.3f},{c['ci95'][1]:+.3f}] p_holm {c['p_holm']:.3g}")
    print("\nretrieval (15-chunk context):")
    for corpus, d in report["retrieval"].items():
        for name, v in d.items():
            print(f"{corpus:<13}{name:<18}{v}")


if __name__ == "__main__":
    main()
