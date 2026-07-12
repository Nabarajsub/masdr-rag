#!/usr/bin/env python3
"""Phase 2a (CPU / login node): parse scraped DOT PDFs into chunks.

Reads ``data/<dot>/manifest.jsonl``, extracts text per PDF with PyMuPDF,
splits into ~1000-char chunks (RecursiveCharacterTextSplitter, 100 overlap --
the same splitter the WYDOT ``local_ingest.py`` pipeline uses), tags each chunk
with section / page / provisional metadata, and writes:

  data/<dot>/derived/chunks.parquet   one row per chunk
  data/<dot>/derived/docs.parquet     one row per document (feeds the Qwen
                                      content classifier in classify_embed.py)

Chunk text is the document content only -- no metadata header is prepended, so
the dilution measurement reflects the documents themselves, not boilerplate.

Usage:
    python -m graph_processing.multi_dot.parse_chunk --dot cdot
    python -m graph_processing.multi_dot.parse_chunk --dot caltrans
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

REPO = Path(__file__).resolve().parents[2]

SECTION_RE = re.compile(r"(SECTION\s+\d+|DIVISION\s+\d+|CHAPTER\s+\d+)", re.IGNORECASE)
YEAR_RE = re.compile(r"((?:19|20)\d{2})")


def extract_pdf(path: Path, extract_tables: bool = False):
    """Return (full_text, page_spans) where page_spans = [(start,end,page_no)].

    Table extraction is off by default: find_tables() is O(page complexity) and
    pathologically slow on 1000-page specs. Plain get_text() captures table text
    anyway (un-gridded), which is sufficient for the dilution measurement -- the
    simpler WYDOT local_ingest.py path also skips table gridding.
    """
    blocks, spans, cursor = [], [], 0
    with fitz.open(path) as doc:
        for i, page in enumerate(doc):
            txt = page.get_text("text")
            if extract_tables:
                try:
                    for tab in page.find_tables():
                        if tab.extract():
                            txt += "\n" + tab.to_pandas().to_markdown(index=False) + "\n"
                except Exception:
                    pass
            if txt.strip():
                blocks.append(txt)
                spans.append((cursor, cursor + len(txt), i + 1))
                cursor += len(txt) + 1
    return "\n".join(blocks), spans


def page_for(offset: int, spans) -> int:
    for start, end, page in spans:
        if start <= offset <= end:
            return page
    return spans[-1][2] if spans else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dot", required=True)
    ap.add_argument("--chunk-size", type=int, default=1000)
    ap.add_argument("--chunk-overlap", type=int, default=100)
    ap.add_argument("--extract-tables", action="store_true",
                    help="grid tables to markdown (slow on huge PDFs)")
    args = ap.parse_args()

    dot_dir = REPO / "data" / args.dot
    manifest_path = dot_dir / "manifest.jsonl"
    if not manifest_path.exists():
        sys.exit(f"no manifest: {manifest_path}")

    docs = [json.loads(l) for l in manifest_path.read_text().splitlines() if l.strip()]
    docs = [d for d in docs if d.get("status") == "downloaded"]
    print(f"[{args.dot}] {len(docs)} documents in manifest")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
    )

    chunk_rows, doc_rows = [], []
    for di, d in enumerate(docs):
        pdf_path = REPO / d["local_path"]
        if not pdf_path.exists():
            print(f"  [miss] {pdf_path}")
            continue
        try:
            full_text, spans = extract_pdf(pdf_path, args.extract_tables)
        except Exception as e:
            print(f"  [parse-fail] {d['filename']}: {e}")
            continue
        if len(full_text.strip()) < 200:
            print(f"  [empty] {d['filename']} (scanned/no text)")
            continue

        first_page = full_text[:6000]
        ym = YEAR_RE.search(d["filename"]) or YEAR_RE.search(first_page)
        year = int(ym.group(1)) if ym else None

        pieces = splitter.split_text(full_text)
        current_section = "General"
        n_doc_chunks = 0
        for ci, piece in enumerate(pieces):
            if len(piece.strip()) < 40:
                continue
            sm = SECTION_RE.search(piece)
            if sm:
                current_section = sm.group(1).upper()
            offset = full_text.find(piece)
            chunk_rows.append({
                "id": f"{args.dot}::{d['filename']}::{ci}",
                "dot": args.dot,
                "source": d["filename"],
                "document_series": d["document_series"],   # provisional
                "title": d["filename"].rsplit(".", 1)[0],
                "year": year,
                "section": current_section,
                "page": page_for(offset, spans) if offset >= 0 else 0,
                "text": piece,
            })
            n_doc_chunks += 1

        doc_rows.append({
            "dot": args.dot,
            "source": d["filename"],
            "document_series": d["document_series"],
            "year": year,
            "n_pages": len(spans),
            "n_chunks": n_doc_chunks,
            "url": d["url"],
            "first_page_text": first_page,
        })
        if (di + 1) % 25 == 0:
            print(f"  parsed {di+1}/{len(docs)} docs, {len(chunk_rows)} chunks")

    derived = dot_dir / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    chunks_df = pd.DataFrame(chunk_rows)
    docs_df = pd.DataFrame(doc_rows)
    chunks_df.to_parquet(derived / "chunks.parquet", index=False)
    docs_df.to_parquet(derived / "docs.parquet", index=False)

    print(f"\n[{args.dot}] DONE")
    print(f"  documents parsed : {len(docs_df)}")
    print(f"  chunks           : {len(chunks_df)}")
    print(f"  -> {derived/'chunks.parquet'}")
    print(f"  -> {derived/'docs.parquet'}")
    if len(chunks_df):
        print(f"\n[{args.dot}] provisional document_series distribution:")
        for series, cnt in chunks_df["document_series"].value_counts().items():
            ndocs = docs_df[docs_df.document_series == series].shape[0]
            print(f"  {series:22s} {ndocs:4d} docs  {cnt:7d} chunks")


if __name__ == "__main__":
    main()
