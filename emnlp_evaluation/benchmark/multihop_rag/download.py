"""
Download MultiHop-RAG corpus + queries.

The benchmark is published in three forms:
  1. HuggingFace dataset `yixuantt/MultiHopRAG`
  2. GitHub raw JSON files
  3. Manually-placed files (already on disk)

Run this from a *login node* with internet access — compute nodes on ARCC
typically have no outbound network. Files land in
emnlp_evaluation/benchmark/multihop_rag/data/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# Raw GitHub mirror (yixuantt/MultiHop-RAG main branch).
_RAW_FILES = {
    "corpus.json": "https://raw.githubusercontent.com/yixuantt/MultiHop-RAG/main/dataset/corpus.json",
    "MultiHopRAG.json": "https://raw.githubusercontent.com/yixuantt/MultiHop-RAG/main/dataset/MultiHopRAG.json",
}


def _download(url: str, dest: Path):
    print(f"[download] {url} -> {dest}", flush=True)
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        f.write(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("github", "hf"), default="github")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.mode == "github":
        for name, url in _RAW_FILES.items():
            dest = DATA_DIR / name
            if dest.exists() and not args.force:
                print(f"[download] {dest.name} already exists, skipping")
                continue
            _download(url, dest)
    else:
        try:
            from datasets import load_dataset
        except ImportError:
            print("Install `datasets` for HF mode.", file=sys.stderr); sys.exit(1)
        ds = load_dataset("yixuantt/MultiHopRAG")
        (DATA_DIR / "MultiHopRAG_hf.jsonl").write_text(
            "\n".join(json.dumps(r) for r in ds["train"])
        )
        print(f"[download] wrote {DATA_DIR / 'MultiHopRAG_hf.jsonl'}")

    # quick sanity
    for name in _RAW_FILES:
        p = DATA_DIR / name
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            print(f"  {name}: {len(data)} records")


if __name__ == "__main__":
    main()
