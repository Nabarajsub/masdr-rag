"""
Local BGE-M3 embedder via sentence-transformers.

BGE-M3 is multilingual, supports up to 8192 input tokens, and produces 1024-dim
dense vectors. We return L2-normalized vectors so a Neo4j vector index with
cosine similarity behaves like dot product.
"""
from __future__ import annotations

import os
from typing import List

import numpy as np


def _resolve_snapshot(model_path: str) -> str:
    snapshots_dir = os.path.join(model_path, "snapshots")
    if os.path.isdir(snapshots_dir):
        snaps = sorted(os.listdir(snapshots_dir))
        if snaps:
            return os.path.join(snapshots_dir, snaps[-1])
    return model_path


class BGEM3Embedder:
    name = "bge_m3"
    dim = 1024

    def __init__(self, model_path: str, *, device: str = "auto", batch_size: int = 32):
        from sentence_transformers import SentenceTransformer
        import torch

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        resolved = _resolve_snapshot(model_path)
        print(f"[bge-m3] loading from {resolved} on {device}", flush=True)
        self.model = SentenceTransformer(resolved, device=device)
        self.batch_size = batch_size

    def embed_query(self, text: str) -> List[float]:
        vec = self.model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False,
        )[0]
        return vec.astype(np.float32).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 1000,
        )
        return [v.astype(np.float32).tolist() for v in vecs]
