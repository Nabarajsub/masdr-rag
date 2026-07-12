#!/usr/bin/env python3
"""Phase 2b (GPU): classify document_series with Qwen, embed chunks with BGE-M3.

Input : data/<dot>/derived/chunks.parquet  (+ docs.parquet for classification)
Output: data/<dot>/derived/<dot>.meta.parquet    -- ingest-ready
        data/<dot>/derived/<dot>.embeddings.npy  -- (n_chunks, 1024) float32

The Qwen content classifier refines the scraper's provisional document_series
(it mostly rescues the 'General' tail). For WYDOT, document_series already came
authoritatively from Neo4j -- pass --skip-classify to embed only.

Run on a GPU node via slurm/run_multidot_embed.sbatch.

Usage:
    python -m graph_processing.multi_dot.classify_embed --dot cdot
    python -m graph_processing.multi_dot.classify_embed --dot wydot --skip-classify
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "graph_processing"))

from emnlp_evaluation.configs import oss_config as cfg          # noqa: E402
from emnlp_evaluation.embeddings.bge_m3 import BGEM3Embedder      # noqa: E402

TAXONOMY = [
    "Standard Specs", "Standard Plans", "Construction Manual", "Materials Testing",
    "Design Manual", "Bridge Program", "Traffic & Safety", "STIP",
    "Annual Reports", "General",
]
TAXONOMY_DESC = """\
- Standard Specs: standard specifications for road and bridge construction, special provisions
- Standard Plans: standard drawings / standard plans (also called M&S Standards)
- Construction Manual: construction administration manuals, construction bulletins
- Materials Testing: materials manuals, test methods, laboratory / field testing procedures
- Design Manual: highway / roadway / pavement / drainage / geometric design manuals and guides
- Bridge Program: bridge design manuals, structures design, memos to designers
- Traffic & Safety: traffic manuals, MUTCD, signing, signals, striping, safety, crash data
- STIP: transportation improvement programs, programming documents, long-range plans
- Annual Reports: annual reports, performance reports, fact books
- General: anything that does not clearly fit a category above"""


def classify_docs(docs_df: pd.DataFrame) -> dict[str, str]:
    """Return {source_filename: refined document_series} using Qwen2.5-7B."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[qwen] loading {cfg.QWEN_PATH}", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg.QWEN_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.QWEN_PATH, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    model.eval()

    lower = {t.lower(): t for t in TAXONOMY}
    mapping: dict[str, str] = {}
    for i, row in enumerate(docs_df.itertuples(index=False)):
        prompt = (
            "Classify this state Department of Transportation document into "
            "exactly ONE category.\n\nCategories:\n" + TAXONOMY_DESC +
            f"\n\nFilename: {row.source}\n\nFirst-page excerpt:\n"
            f"{(row.first_page_text or '')[:2500]}\n\n"
            "Respond with ONLY the category name, exactly as written above."
        )
        msgs = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=24, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        ans = tok.decode(out[0][inputs.input_ids.shape[1]:],
                         skip_special_tokens=True).strip().lower()
        label = next((lower[k] for k in lower if k in ans), None)
        mapping[row.source] = label or row.document_series
        if (i + 1) % 25 == 0:
            print(f"  classified {i+1}/{len(docs_df)}", flush=True)

    del model
    torch.cuda.empty_cache()
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dot", required=True)
    ap.add_argument("--skip-classify", action="store_true",
                    help="keep document_series as-is (use for WYDOT)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-seq-len", type=int, default=512,
                    help="cap BGE-M3 input tokens (default 512 fits T4 16GB; "
                         "chunks are ~250 tokens so this rarely truncates)")
    ap.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="bf16",
                    help="model dtype; bf16 on Ampere (A30/L40S/H100) is "
                         "~3-4x faster than fp32 with negligible cosine drift")
    args = ap.parse_args()

    derived = REPO / "data" / args.dot / "derived"
    chunks = pd.read_parquet(derived / "chunks.parquet")
    print(f"[{args.dot}] {len(chunks)} chunks loaded")

    if not args.skip_classify:
        docs = pd.read_parquet(derived / "docs.parquet")
        mapping = classify_docs(docs)
        before = chunks["document_series"].copy()
        chunks["document_series"] = chunks["source"].map(mapping).fillna(
            chunks["document_series"])
        moved = int((before != chunks["document_series"]).sum())
        print(f"[{args.dot}] Qwen re-categorised {moved}/{len(chunks)} chunks")

    print(f"[bge-m3] embedding {len(chunks)} chunks "
          f"(batch={args.batch_size}, max_seq_len={args.max_seq_len}, "
          f"dtype={args.dtype})", flush=True)
    embedder = BGEM3Embedder(cfg.BGE_M3_PATH, batch_size=args.batch_size)
    embedder.model.max_seq_length = args.max_seq_len
    if args.dtype != "fp32":
        import torch
        dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16
        embedder.model = embedder.model.to(dt)
        print(f"[bge-m3] cast to {args.dtype}", flush=True)
    vecs = embedder.embed_documents(chunks["text"].astype(str).tolist())
    embs = np.asarray(vecs, dtype=np.float32)
    assert embs.shape[0] == len(chunks), (embs.shape, len(chunks))

    meta_cols = ["id", "text", "title", "year", "section",
                 "document_series", "dot", "source", "page"]
    meta = chunks[meta_cols].copy()
    meta_path = derived / f"{args.dot}.meta.parquet"
    emb_path = derived / f"{args.dot}.embeddings.npy"
    meta.to_parquet(meta_path, index=False)
    np.save(emb_path, embs)

    print(f"\n[{args.dot}] DONE  meta={meta_path}  emb={emb_path}  dim={embs.shape[1]}")
    print(f"[{args.dot}] final document_series distribution:")
    for series, cnt in meta["document_series"].value_counts().items():
        print(f"  {series:22s} {cnt:7d} chunks")


if __name__ == "__main__":
    main()
