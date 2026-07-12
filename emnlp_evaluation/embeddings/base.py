"""Embedder protocol — every backend returns L2-normalized vectors."""
from __future__ import annotations

from typing import List, Protocol


class Embedder(Protocol):
    name: str
    dim: int

    def embed_query(self, text: str) -> List[float]: ...
    def embed_documents(self, texts: List[str]) -> List[List[float]]: ...
