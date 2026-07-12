"""Gemini embedder wrapper to match the Embedder protocol."""
from __future__ import annotations

from typing import List


class GeminiEmbedder:
    name = "gemini"
    dim = 768  # gemini-embedding-001 returns 768-dim by default

    def __init__(self, api_key: str, model_name: str = "gemini-embedding-001"):
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        self._impl = GoogleGenerativeAIEmbeddings(
            model=f"models/{model_name}", google_api_key=api_key,
        )

    def embed_query(self, text: str) -> List[float]:
        return self._impl.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._impl.embed_documents(texts)
