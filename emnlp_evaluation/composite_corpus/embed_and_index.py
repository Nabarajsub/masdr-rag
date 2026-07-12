"""
Read the assembled parquet, embed every chunk with BGE-M3, save:
  <out>.embeddings.npy   (float32, [N, dim], L2-normalized)
  <out>.meta.parquet     (id, doc_id, source_type, title, text, idx)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO.parent))

from emnlp_evaluation.embeddings import get_embedder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(Path(__file__).resolve().parent / "data" / "corpus.parquet"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "data" / "composite"))
    ap.add_argument("--batch-size", type=int, default=16,
                    help="Batch size for embedder. Lowered to 16 to fit "
                         "long financial / report inputs on A30 (24 GB).")
    ap.add_argument("--max-chars", type=int, default=4000,
                    help="Truncate each chunk to this many characters before "
                         "embedding (BGE-M3 has 8k token limit; ~4k chars is safe).")
    args = ap.parse_args()

    df = pd.read_parquet(args.corpus).reset_index(drop=True)
    print(f"[embed] {len(df)} chunks from {args.corpus}", flush=True)

    embedder = get_embedder("bge_m3")
    texts = df["text"].astype(str).str[:args.max_chars].tolist()

    t0 = time.time()
    vecs = []
    bs = args.batch_size
    for i in range(0, len(texts), bs):
        batch = texts[i : i + bs]
        chunk = embedder.embed_documents(batch)
        vecs.extend(chunk)
        if (i // bs) % 20 == 0:
            done = i + len(batch)
            rate = done / max(time.time() - t0, 1e-6)
            print(f"[embed] {done}/{len(texts)} ({rate:.1f}/s)", flush=True)

    arr = np.asarray(vecs, dtype=np.float32)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(f"{out}.embeddings.npy", arr)
    df.to_parquet(f"{out}.meta.parquet", index=False)
    print(f"[embed] wrote {out}.embeddings.npy [{arr.shape}] and {out}.meta.parquet")


if __name__ == "__main__":
    main()
