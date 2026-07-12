"""GraphRAG-style ablation on the unified-FAISS WYDOT index.

The standard GraphRAG pipeline (Edge et al., 2024):
  1. Extract entities + relationships from each chunk
  2. Build entity graph, run community detection (Leiden)
  3. Generate LLM summaries per community
  4. At query time: retrieve community summaries, then drill down to chunks

We use a lightweight analog that captures the structural argument
without requiring an entity-extraction pass over 88k chunks:

  1. K-means cluster the BGE-M3 chunk embeddings (k=24) — communities
  2. For each cluster, generate a 200-word Qwen summary from the 5
     most-central chunks
  3. At query time: rank communities by query-summary similarity, take
     the top-3 communities' chunks, then re-rank to top-15

Compares vs:
  - Monolithic FAISS (existing baseline)
  - Per-scope retrieval (existing scoped baselines)

Outputs:
    results/wydot_qwen_graphrag.jsonl
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

REPO = Path("<DATA_ROOT>")
WYDOT_PREFIX = REPO / "emnlp_evaluation/composite_corpus/data/wydot"
SUITE_PATH = REPO / "evaluation/test_suite_200.json"
OUT_PATH = REPO / "emnlp_evaluation/results/wydot_qwen_graphrag.jsonl"
SUMMARY_CACHE = REPO / "emnlp_evaluation/results/wydot_graphrag_summaries.json"

K_COMMUNITIES = 24
TOPK_COMMUNITIES = 3
TOPK_CHUNKS = 15
SUMMARY_WORDS = 200
CENTRAL_PER_COMMUNITY = 5


def _load_embeds_meta():
    emb = np.load(f"{WYDOT_PREFIX}.embeddings.npy").astype(np.float32)
    meta = pd.read_parquet(f"{WYDOT_PREFIX}.meta.parquet").reset_index(drop=True)
    return emb, meta


def _build_or_load_communities(emb, meta):
    if SUMMARY_CACHE.exists():
        cached = json.loads(SUMMARY_CACHE.read_text())
        print(f"[gr] loading cached {len(cached['summaries'])} community summaries")
        return np.array(cached["assignments"]), cached["summaries"], np.array(cached["centroids"])

    print(f"[gr] k-means clustering {len(emb)} chunks into k={K_COMMUNITIES} communities")
    km = KMeans(n_clusters=K_COMMUNITIES, random_state=0, n_init=4)
    assignments = km.fit_predict(emb)
    centroids = km.cluster_centers_.astype(np.float32)

    # Pick top-5 most-central chunks per community for the summary prompt.
    print("[gr] selecting central chunks per community")
    central_idxs = {}
    for c in range(K_COMMUNITIES):
        mask = assignments == c
        if not mask.any():
            continue
        sub_idx = np.where(mask)[0]
        d = np.linalg.norm(emb[sub_idx] - centroids[c], axis=1)
        order = np.argsort(d)[:CENTRAL_PER_COMMUNITY]
        central_idxs[c] = sub_idx[order].tolist()

    # Generate one summary per community via Qwen.
    print("[gr] generating community summaries via Qwen-7B")
    from emnlp_evaluation.llm_providers import get_provider
    llm = get_provider("qwen")
    summaries = {}
    for c, idxs in central_idxs.items():
        snippets = []
        for i in idxs:
            row = meta.iloc[i]
            snippets.append(
                f"- [{(row.get('title') or '')[:80]}] {(row.get('text') or '')[:600]}"
            )
        prompt = (
            f"Read the {len(snippets)} document excerpts below from a Wyoming "
            f"Department of Transportation knowledge base. Write a single "
            f"{SUMMARY_WORDS}-word summary describing the topic / scope of "
            f"this collection. Cover names, document types, and the dominant "
            f"subject. Do not invent details.\n\n"
            f"Excerpts:\n" + "\n".join(snippets) +
            f"\n\nSummary ({SUMMARY_WORDS} words):"
        )
        res = llm.generate([{"role": "user", "content": prompt}],
                           max_new_tokens=400, temperature=0.2)
        summaries[c] = res.text.strip()
        print(f"  community {c:>2}: {summaries[c][:80]}...", flush=True)

    SUMMARY_CACHE.write_text(json.dumps({
        "k": K_COMMUNITIES,
        "assignments": assignments.tolist(),
        "centroids": centroids.tolist(),
        "summaries": {str(k): v for k, v in summaries.items()},
        "central_idxs": {str(k): v for k, v in central_idxs.items()},
    }))
    print(f"[gr] cached to {SUMMARY_CACHE}")
    return assignments, {str(k): v for k, v in summaries.items()}, centroids


def main():
    import sys
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO.parent))
    from emnlp_evaluation.llm_providers import get_provider
    from emnlp_evaluation.embeddings import get_embedder

    emb, meta = _load_embeds_meta()
    print(f"[gr] WYDOT: {len(emb)} chunks loaded")
    assignments, summaries, centroids = _build_or_load_communities(emb, meta)

    # Build summary embeddings for query-summary similarity.
    embedder = get_embedder("bge_m3")
    print("[gr] embedding community summaries")
    sum_texts = [summaries[str(c)] for c in range(K_COMMUNITIES) if str(c) in summaries]
    sum_ids = [c for c in range(K_COMMUNITIES) if str(c) in summaries]
    sum_embs = np.asarray(embedder.embed_documents(sum_texts), dtype=np.float32)
    sum_embs /= np.linalg.norm(sum_embs, axis=1, keepdims=True)

    llm = get_provider("qwen")
    suite = json.load(open(SUITE_PATH))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if OUT_PATH.exists():
        for line in OUT_PATH.open():
            try:
                r = json.loads(line)
                if "error" not in r:
                    done.add((r["query_id"], r["system"]))
            except json.JSONDecodeError:
                continue

    with OUT_PATH.open("a") as f:
        for q in suite:
            qid = q.get("id") or q.get("query_id")
            if (qid, "graphrag") in done:
                continue
            t0 = time.time()
            try:
                # 1. Rank communities by query-summary similarity.
                q_vec = np.asarray(embedder.embed_query(q["query"]),
                                   dtype=np.float32)
                q_vec /= np.linalg.norm(q_vec)
                sims = sum_embs @ q_vec
                top_c = [sum_ids[i] for i in np.argsort(-sims)[:TOPK_COMMUNITIES]]
                # 2. Restrict chunks to selected communities; rank by cosine to query.
                in_comm = np.isin(assignments, top_c)
                sub_idx = np.where(in_comm)[0]
                sub_emb = emb[sub_idx]
                chunk_sims = sub_emb @ q_vec
                top = np.argsort(-chunk_sims)[:TOPK_CHUNKS]
                chunks = []
                for j in top:
                    row = meta.iloc[int(sub_idx[j])]
                    chunks.append({
                        "id": str(row.get("id", "")),
                        "text": str(row.get("text", "")),
                        "title": str(row.get("title", "")),
                        "source": str(row.get("source_type", "")),
                        "score": float(chunk_sims[j]),
                    })
                # 3. Synthesise with Qwen.
                ctx = "\n---\n".join(
                    f"[SOURCE {i+1}: {c.get('title','')}, src={c.get('source','')}]\n{c['text'][:1500]}"
                    for i, c in enumerate(chunks)
                )
                prompt = (
                    f"Answer the question using ONLY the retrieved sources. "
                    f"Cite as [Source 1], [Source 2], etc.\n\n"
                    f"Retrieved sources:\n{ctx}\n\nQuestion: {q['query']}\nAnswer:"
                )
                res = llm.generate([{"role": "user", "content": prompt}],
                                   max_new_tokens=1024, temperature=0.2)
                rec = {
                    "query_id": qid,
                    "query": q["query"],
                    "system": "graphrag",
                    "answer": res.text,
                    "reference_answer": q.get("reference_answer", ""),
                    "n_chunks": len(chunks),
                    "chunk_ids": [c["id"] for c in chunks],
                    "chunk_series": [c["title"] for c in chunks],
                    "chunk_sources": [c["source"] for c in chunks],
                    "selected_communities": top_c,
                    "category": q.get("category"),
                    "relevant_title": q.get("relevant_title"),
                    "prompt_tokens": res.prompt_tokens,
                    "completion_tokens": res.completion_tokens,
                    "llm_calls": 1,
                    "wall_time_s": time.time() - t0,
                    "llm": "qwen",
                    "embedder": "bge_m3",
                    "benchmark": "wydot",
                }
            except Exception as e:
                rec = {"query_id": qid, "system": "graphrag",
                       "error": f"{type(e).__name__}: {e}"}
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(f"[{qid:>6}] [graphrag] {time.time()-t0:5.1f}s "
                  f"chunks={rec.get('n_chunks','-')} comms={rec.get('selected_communities','-')}",
                  flush=True)

    print(f"[gr] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
