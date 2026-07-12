"""Pick an embedder by name."""
from __future__ import annotations

import os

from ..configs import oss_config as cfg
from .base import Embedder


_CACHE: dict = {}


def get_embedder(name: str, **overrides) -> Embedder:
    key = (name, tuple(sorted(overrides.items())))
    if key in _CACHE:
        return _CACHE[key]

    name = name.lower()
    if name == "gemini":
        from .gemini_embed import GeminiEmbedder
        api_key = overrides.get("api_key") or os.getenv("GEMINI_API_KEY_V2") or os.getenv("GEMINI_API_KEY", "")
        embedder = GeminiEmbedder(api_key=api_key)
    elif name == "bge_m3":
        from .bge_m3 import BGEM3Embedder
        embedder = BGEM3Embedder(model_path=overrides.get("model_path", cfg.BGE_M3_PATH))
    else:
        raise ValueError(f"Unknown embedder: {name}. Supported: {cfg.SUPPORTED_EMBEDDERS}")

    _CACHE[key] = embedder
    return embedder
