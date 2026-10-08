"""Run run_wydot_oss with Gemini *query* embeddings served from a cache.

The installed langchain_google_genai no longer imports in the wydot_qwen env, so
the 200 suite queries were embedded once (gemini-embedding-001, embed_query,
3072-d -- identical to the production index; stored chunk re-embeds give cosine
1.0) into router/artifacts/wydot_queries_gemini_embeddings.npz. Systems that only
embed the user query (e.g. soft_scoped) can then run unchanged.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from emnlp_evaluation.runners import run_wydot_oss as R

_CACHE = Path(__file__).resolve().parents[1] / "router" / "artifacts" / "wydot_queries_gemini_embeddings.npz"


class CachedGeminiEmbedder:
    name = "gemini"
    dim = 3072

    def __init__(self):
        z = np.load(_CACHE, allow_pickle=True)
        self._map = {str(t): v.astype(np.float64).tolist() for t, v in zip(z["texts"], z["vecs"])}

    def embed_query(self, text: str):
        if text not in self._map:
            raise KeyError(f"query not in Gemini cache: {text[:80]!r}")
        return self._map[text]

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]


_orig = R.get_embedder
R.get_embedder = lambda name, *a, **k: CachedGeminiEmbedder() if name == "gemini" else _orig(name, *a, **k)

if __name__ == "__main__":
    R.main()
