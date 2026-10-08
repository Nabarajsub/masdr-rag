"""RQ3: are the discovered scopes stable, and is k chosen without hand labels?

  * seed stability at k=11: 10 k-means seeds -> pairwise ARI between partitions,
    and NMI / chunk purity vs the hand document_series (mean, SD)
  * k sweep 5..20: NMI vs hand labels (what discover_scopes.py used to pick k)
    next to two LABEL-FREE criteria (silhouette, Calinski-Harabasz) -> which k
    would a tenant with no metadata actually pick, and how good is it?
  * check the shipped `discovered_scope` column matches the seed-42 k=11 run
"""
from __future__ import annotations
import argparse, itertools, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score, calinski_harabasz_score)
from graph_processing.multi_dot.discover_scopes import doc_pool, purity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    meta = pd.read_parquet(a.prefix + ".meta.parquet")
    emb = np.load(a.prefix + ".embeddings.npy", mmap_mode="r")
    docvecs, codes, uniques = doc_pool(emb, meta)
    doc_series = (meta.groupby("source")["document_series"].agg(lambda s: s.mode().iat[0])
                  .reindex(uniques).to_numpy())
    chunk_series = meta["document_series"].to_numpy()
    rep = {"n_docs": int(len(uniques)), "n_chunks": int(len(meta))}

    def fit(k, seed):
        return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(docvecs).labels_

    # seed stability at k=11
    parts = {s: fit(11, s) for s in range(10)}
    aris = [adjusted_rand_score(parts[i], parts[j]) for i, j in itertools.combinations(parts, 2)]
    nmis = [normalized_mutual_info_score(doc_series, p) for p in parts.values()]
    purs = [purity(chunk_series, p[codes]) for p in parts.values()]
    rep["k11_seed_stability"] = {"pairwise_ARI_mean": float(np.mean(aris)), "pairwise_ARI_min": float(np.min(aris)),
                                 "doc_NMI_mean": float(np.mean(nmis)), "doc_NMI_sd": float(np.std(nmis)),
                                 "chunk_purity_mean": float(np.mean(purs)), "chunk_purity_sd": float(np.std(purs))}
    # shipped column vs seed-42 partition
    if "discovered_scope" in meta.columns:
        p42 = fit(11, 42)[codes]
        rep["shipped_column_vs_seed42_ARI"] = float(adjusted_rand_score(meta["discovered_scope"].astype(str), p42))

    # k sweep: hand-label criterion vs label-free criteria
    sweep = []
    for k in range(5, 21):
        lab = fit(k, 42)
        sweep.append({"k": k,
                      "doc_NMI_hand": float(normalized_mutual_info_score(doc_series, lab)),
                      "chunk_purity_hand": float(purity(chunk_series, lab[codes])),
                      "silhouette": float(silhouette_score(docvecs, lab, metric="cosine")),
                      "calinski_harabasz": float(calinski_harabasz_score(docvecs, lab))})
        print(sweep[-1], flush=True)
    rep["k_sweep"] = sweep
    best = lambda key: max(sweep, key=lambda r: r[key])
    rep["k_by_hand_NMI"] = best("doc_NMI_hand")["k"]
    rep["k_by_silhouette"] = best("silhouette")["k"]
    rep["k_by_CH"] = best("calinski_harabasz")["k"]
    Path(a.out).write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k != "k_sweep"}, indent=1))


if __name__ == "__main__":
    main()
