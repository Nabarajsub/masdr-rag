"""
Qwen2.5-7B-Instruct via HuggingFace transformers.

Qwen2.5's official chat template handles tool calling natively: pass tools to
`apply_chat_template` and the model emits `<tool_call>{"name": ..., "arguments": ...}</tool_call>`
blocks that we parse back out.

Reference: https://qwen.readthedocs.io/en/latest/framework/function_call.html
"""
from __future__ import annotations

import os

import json
import re
import time
from typing import Any, Dict, List, Optional

import torch

from .base import GenerationResult, ToolCall, ToolSpec

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Coerce inbound messages to the structure Qwen expects.

    Tool results (role='tool') need a `name` field; otherwise pass through.
    """
    out: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        if role == "tool":
            out.append({
                "role": "tool",
                "name": m.get("name", "unknown_tool"),
                "content": m.get("content", ""),
            })
        elif role == "assistant" and m.get("tool_calls"):
            out.append({
                "role": "assistant",
                "content": m.get("content", "") or "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": tc["name"] if isinstance(tc, dict) else tc.name,
                            "arguments": (
                                tc["arguments"] if isinstance(tc, dict) else tc.arguments
                            ),
                        },
                    }
                    for tc in m["tool_calls"]
                ],
            })
        else:
            out.append({"role": role, "content": m.get("content", "")})
    return out


class QwenProvider:
    name = "qwen"

    def __init__(
        self,
        model_path: str,
        *,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        trust_remote_code: bool = True,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[dtype]

        print(f"[qwen] loading tokenizer from {model_path}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        print(f"[qwen] loading model ({dtype}, device_map={device_map})", flush=True)
        t0 = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()
        print(f"[qwen] loaded in {time.time()-t0:.1f}s", flush=True)

    # ── public API ──────────────────────────────────────────────────────────

    def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        max_new_tokens: int = 1024,
        temperature: float = 0.2,
        stop: Optional[List[str]] = None,
    ) -> GenerationResult:
        # GEN_TEMPERATURE overrides sampled (temperature > 0) generation only, so a
        # run can be made greedy without touching the already-greedy router/judge calls.
        if temperature > 0 and "GEN_TEMPERATURE" in os.environ:
            temperature = float(os.environ["GEN_TEMPERATURE"])
        return self._generate(
            messages,
            tools=None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    def generate_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[ToolSpec],
        *,
        max_new_tokens: int = 1024,
        temperature: float = 0.2,
        tool_choice: str = "auto",
    ) -> GenerationResult:
        return self._generate(
            messages,
            tools=tools,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    # ── internals ───────────────────────────────────────────────────────────

    def _generate(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[ToolSpec]],
        max_new_tokens: int,
        temperature: float,
    ) -> GenerationResult:
        norm = _normalize_messages(messages)
        tool_payload = (
            [t.to_openai() for t in tools] if tools else None
        )

        chat_text = self.tokenizer.apply_chat_template(
            norm,
            tools=tool_payload,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.tokenizer(
            chat_text, return_tensors="pt", add_special_tokens=False
        ).to(self.model.device)
        prompt_tokens = int(inputs["input_ids"].shape[1])

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        new_ids = output_ids[0, prompt_tokens:]
        completion_tokens = int(new_ids.shape[0])
        raw_text = self.tokenizer.decode(new_ids, skip_special_tokens=True)

        tool_calls, clean_text = _parse_tool_calls(raw_text)

        return GenerationResult(
            text=clean_text.strip(),
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason="tool_call" if tool_calls else "stop",
            raw=raw_text,
        )


def _parse_tool_calls(text: str):
    """Extract `<tool_call>...</tool_call>` blocks from a Qwen response."""
    calls: List[ToolCall] = []
    for m in _TOOL_CALL_RE.finditer(text):
        blob = m.group(1).strip()
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            # Qwen sometimes emits trailing commas or single quotes; try a soft fix.
            try:
                obj = json.loads(blob.replace("'", '"'))
            except json.JSONDecodeError:
                continue
        name = obj.get("name")
        args = obj.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw": args}
        if name:
            calls.append(ToolCall(name=name, arguments=args or {}))
    clean = _TOOL_CALL_RE.sub("", text)
    return calls, clean
