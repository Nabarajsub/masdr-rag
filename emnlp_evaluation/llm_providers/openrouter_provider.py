"""
OpenRouter LLM provider (HTTP, no SDK dependency).

OpenRouter exposes an OpenAI-compatible /chat/completions endpoint that fronts
~200 models. We use it to run third-backbone experiments (Claude-Sonnet, GPT,
DeepSeek, Llama-70B) without paying separate vendor onboarding cost.

Set OPENROUTER_API_KEY in env. Pass the OpenRouter model slug (e.g.,
"anthropic/claude-sonnet-4.5", "openai/gpt-5-mini", "deepseek/deepseek-chat")
as model_name.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import requests

from .base import GenerationResult, ToolCall, ToolSpec


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """OpenAI-strict round-trip: assistant tool_calls and the subsequent tool
    messages must agree on tool_call_id. Our orchestrator only threads tool
    names through, so we synthesise deterministic ids per (assistant_turn, name)
    and queue them so tool responses pop them in FIFO order — which is the
    order the orchestrator appends responses."""
    out: List[Dict[str, Any]] = []
    # FIFO queue of pending tool_call_ids per tool name, from the most recent
    # assistant turn.
    pending: Dict[str, list] = {}

    for m in messages:
        role = m.get("role")
        if role == "tool":
            name = m.get("name", "unknown_tool")
            explicit = m.get("tool_call_id")
            if explicit:
                tcid = explicit
            elif pending.get(name):
                tcid = pending[name].pop(0)
            else:
                tcid = f"call_{name}_orphan"
            out.append({
                "role": "tool",
                "tool_call_id": tcid,
                "name": name,
                "content": m.get("content", "") or "",
            })
        elif role == "assistant" and m.get("tool_calls"):
            converted_calls = []
            pending = {}
            for i, tc in enumerate(m["tool_calls"]):
                name = tc["name"] if isinstance(tc, dict) else tc.name
                explicit = tc.get("id") if isinstance(tc, dict) else getattr(tc, "call_id", None)
                tcid = explicit or f"call_{name}_{i}"
                pending.setdefault(name, []).append(tcid)
                args = tc["arguments"] if isinstance(tc, dict) else tc.arguments
                args_str = args if isinstance(args, str) else json.dumps(args)
                converted_calls.append({
                    "id": tcid,
                    "type": "function",
                    "function": {"name": name, "arguments": args_str},
                })
            out.append({
                "role": "assistant",
                "content": m.get("content", "") or "",
                "tool_calls": converted_calls,
            })
        else:
            out.append({"role": role, "content": m.get("content", "") or ""})
    return out


class OpenRouterProvider:
    name = "openrouter"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model_name: str = "anthropic/claude-sonnet-4.5",
        timeout: int = 120,
        max_retries: int = 3,
        site_url: str = "https://github.com/wydot-rag-emnlp2026",
        app_name: str = "wydot-emnlp-eval",
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.site_url = site_url
        self.app_name = app_name
        # Naive per-process cost ledger so callers can see how much they spent.
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        print(f"[openrouter] model={model_name}", flush=True)

    # ── public API ──────────────────────────────────────────────────────────

    def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        max_new_tokens: int = 1024,
        temperature: float = 0.2,
        stop: Optional[List[str]] = None,
    ) -> GenerationResult:
        return self._call(messages, tools=None,
                          max_new_tokens=max_new_tokens,
                          temperature=temperature, stop=stop)

    def generate_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[ToolSpec],
        *,
        max_new_tokens: int = 1024,
        temperature: float = 0.2,
        tool_choice: str = "auto",
    ) -> GenerationResult:
        return self._call(messages, tools=tools,
                          max_new_tokens=max_new_tokens,
                          temperature=temperature,
                          tool_choice=tool_choice)

    def count_tokens(self, text: str) -> int:
        # Cheap upper bound: 1 token per 4 chars. OpenRouter returns real
        # usage in the response; the runners that care use that instead.
        return max(1, len(text) // 4)

    # ── internals ───────────────────────────────────────────────────────────

    def _call(self, messages, *, tools, max_new_tokens, temperature,
              stop=None, tool_choice: str = "auto") -> GenerationResult:
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": _normalize_messages(messages),
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }
        if stop:
            payload["stop"] = stop
        if tools:
            payload["tools"] = [t.to_openai() for t in tools]
            payload["tool_choice"] = tool_choice
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url,
            "X-Title": self.app_name,
        }

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(OPENROUTER_URL, headers=headers,
                                  data=json.dumps(payload), timeout=self.timeout)
                if r.status_code == 429 or r.status_code >= 500:
                    # rate-limit / transient server: backoff
                    raise RuntimeError(f"openrouter {r.status_code}: {r.text[:200]}")
                if r.status_code != 200:
                    raise RuntimeError(f"openrouter {r.status_code}: {r.text[:300]}")
                data = r.json()
                return self._parse(data)
            except Exception as e:
                last_err = e
                wait = 2 ** attempt
                print(f"[openrouter] attempt {attempt+1}/{self.max_retries} "
                      f"failed: {e}; retry in {wait}s", flush=True)
                time.sleep(wait)
        raise RuntimeError(f"openrouter failed after {self.max_retries} tries: {last_err}")

    def _parse(self, data: Dict[str, Any]) -> GenerationResult:
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        text = msg.get("content") or ""
        usage = data.get("usage") or {}
        prompt_t = int(usage.get("prompt_tokens") or 0)
        comp_t = int(usage.get("completion_tokens") or 0)
        self.total_prompt_tokens += prompt_t
        self.total_completion_tokens += comp_t

        tool_calls: List[ToolCall] = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args_str = fn.get("arguments") or "{}"
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else (args_str or {})
            except json.JSONDecodeError:
                args = {"_raw": args_str}
            tool_calls.append(ToolCall(
                name=fn.get("name", ""),
                arguments=args or {},
                call_id=tc.get("id"),
            ))

        finish = choice.get("finish_reason") or ("tool_call" if tool_calls else "stop")
        return GenerationResult(
            text=text.strip(),
            tool_calls=tool_calls,
            prompt_tokens=prompt_t,
            completion_tokens=comp_t,
            finish_reason=finish,
            raw=data,
        )
