"""Pick an LLM provider by name."""
from __future__ import annotations

import os

from ..configs import oss_config as cfg
from .base import LLMProvider


_CACHE: dict = {}


def get_provider(name: str, **overrides) -> LLMProvider:
    """Return a cached provider so the model only loads once per process."""
    key = (name, tuple(sorted(overrides.items())))
    if key in _CACHE:
        return _CACHE[key]

    name = name.lower()
    if name == "gemini":
        from .gemini_provider import GeminiProvider
        api_key = overrides.get("api_key") or os.getenv("GEMINI_API_KEY_V2") or os.getenv("GEMINI_API_KEY", "")
        model = overrides.get("model_name", os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash"))
        provider = GeminiProvider(api_key=api_key, model_name=model)
    elif name == "qwen":
        from .qwen_provider import QwenProvider
        provider = QwenProvider(model_path=overrides.get("model_path", cfg.QWEN_PATH))
    elif name == "llama":
        from .llama_provider import LlamaProvider
        provider = LlamaProvider(model_path=overrides.get("model_path", cfg.LLAMA_PATH))
    elif name in ("openrouter", "claude", "gpt", "deepseek"):
        from .openrouter_provider import OpenRouterProvider
        default_model = {
            "claude":     "anthropic/claude-sonnet-4.5",
            "gpt":        "openai/gpt-5-mini",
            "deepseek":   "deepseek/deepseek-chat",
            "openrouter": overrides.get("model_name") or os.getenv(
                "OPENROUTER_MODEL", "anthropic/claude-sonnet-4.5"),
        }[name]
        model = overrides.get("model_name") or default_model
        provider = OpenRouterProvider(model_name=model)
    else:
        raise ValueError(f"Unknown provider: {name}. Supported: {cfg.SUPPORTED_PROVIDERS} + openrouter/claude/gpt/deepseek")

    _CACHE[key] = provider
    return provider
