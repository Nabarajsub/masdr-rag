"""
Meta-Llama-3-8B-Instruct via HuggingFace transformers.

Llama-3 base chat template does NOT have first-class tool calling, so we use
the same Hermes/Qwen-style `<tool_call>{...}</tool_call>` convention but
inject the tool catalog into the system prompt. The parser is the same one
QwenProvider uses, so all downstream code stays identical.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import torch

from .base import GenerationResult, ToolSpec
from .qwen_provider import _normalize_messages, _parse_tool_calls


_TOOL_SYSTEM_BLOCK = """You have access to the following tools. When a tool is needed, emit a SINGLE block:

<tool_call>{{"name": "<tool name>", "arguments": {{...}}}}</tool_call>

Emit one tool call per turn. After tool results are returned (role=tool), continue reasoning or emit another tool call. When you are done, write the final answer in plain text with no <tool_call> blocks.

Available tools (JSON Schema):
{tool_catalog}
"""


def _format_tool_catalog(tools: List[ToolSpec]) -> str:
    return json.dumps([t.to_openai() for t in tools], indent=2)


def _resolve_snapshot(model_path: str) -> str:
    """If `model_path` points at an HF cache repo dir, pick the latest snapshot."""
    import os
    snapshots_dir = os.path.join(model_path, "snapshots")
    if os.path.isdir(snapshots_dir):
        snaps = sorted(os.listdir(snapshots_dir))
        if snaps:
            return os.path.join(snapshots_dir, snaps[-1])
    return model_path


class LlamaProvider:
    name = "llama"

    def __init__(
        self,
        model_path: str,
        *,
        dtype: str = "bfloat16",
        device_map: str = "auto",
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        resolved = _resolve_snapshot(model_path)
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[dtype]

        print(f"[llama] loading tokenizer from {resolved}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(resolved)
        print(f"[llama] loading model ({dtype}, device_map={device_map})", flush=True)
        t0 = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(
            resolved, torch_dtype=torch_dtype, device_map=device_map,
        )
        self.model.eval()
        print(f"[llama] loaded in {time.time()-t0:.1f}s", flush=True)

    def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        max_new_tokens: int = 1024,
        temperature: float = 0.2,
        stop: Optional[List[str]] = None,
    ) -> GenerationResult:
        return self._generate(messages, tools=None,
                              max_new_tokens=max_new_tokens, temperature=temperature)

    def generate_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[ToolSpec],
        *,
        max_new_tokens: int = 1024,
        temperature: float = 0.2,
        tool_choice: str = "auto",
    ) -> GenerationResult:
        return self._generate(messages, tools=tools,
                              max_new_tokens=max_new_tokens, temperature=temperature)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _generate(self, messages, *, tools, max_new_tokens, temperature) -> GenerationResult:
        norm = _normalize_messages(messages)
        if tools:
            sys_block = _TOOL_SYSTEM_BLOCK.format(tool_catalog=_format_tool_catalog(tools))
            if norm and norm[0]["role"] == "system":
                norm[0]["content"] = sys_block + "\n\n" + norm[0]["content"]
            else:
                norm = [{"role": "system", "content": sys_block}] + norm

        chat_text = self.tokenizer.apply_chat_template(
            norm, add_generation_prompt=True, tokenize=False,
        )
        inputs = self.tokenizer(
            chat_text, return_tensors="pt", add_special_tokens=False
        ).to(self.model.device)
        prompt_tokens = int(inputs["input_ids"].shape[1])

        # Llama tokenizers may lack a pad token.
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        eot = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        eos_ids = [self.tokenizer.eos_token_id]
        if eot and eot != self.tokenizer.unk_token_id and eot not in eos_ids:
            eos_ids.append(eot)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=0.9,
                pad_token_id=pad_id,
                eos_token_id=eos_ids,
            )

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
