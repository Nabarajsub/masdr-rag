"""
Gemini provider — wraps `google.generativeai` so the existing Gemini stack
can be benchmarked through the same interface as Qwen/Llama.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .base import GenerationResult, ToolCall, ToolSpec


def _to_proto_schema(schema: Dict[str, Any]):
    import google.generativeai as genai
    TYPE = {
        "string": genai.protos.Type.STRING,
        "integer": genai.protos.Type.INTEGER,
        "number": genai.protos.Type.NUMBER,
        "boolean": genai.protos.Type.BOOLEAN,
        "object": genai.protos.Type.OBJECT,
        "array": genai.protos.Type.ARRAY,
    }
    t = schema.get("type", "object")
    proto = genai.protos.Schema(type=TYPE[t])
    if "description" in schema:
        proto.description = schema["description"]
    if t == "object":
        for k, v in schema.get("properties", {}).items():
            proto.properties[k] = _to_proto_schema(v)
        for req in schema.get("required", []):
            proto.required.append(req)
    elif t == "array" and "items" in schema:
        proto.items.CopyFrom(_to_proto_schema(schema["items"]))
    return proto


def _to_function_declaration(tool: ToolSpec):
    import google.generativeai as genai
    return genai.protos.FunctionDeclaration(
        name=tool.name,
        description=tool.description,
        parameters=_to_proto_schema(tool.parameters),
    )


def _messages_to_gemini(messages: List[Dict[str, Any]]):
    """Convert OpenAI-style messages to Gemini `contents` parts."""
    import google.generativeai as genai
    contents = []
    for m in messages:
        role = m["role"]
        if role == "system":
            continue  # gemini takes system_instruction separately
        gemini_role = "user" if role in ("user", "tool") else "model"
        if role == "tool":
            part = genai.protos.Part(
                function_response=genai.protos.FunctionResponse(
                    name=m.get("name", "unknown"),
                    response={"content": m.get("content", "")},
                )
            )
            contents.append(genai.protos.Content(role="user", parts=[part]))
        elif role == "assistant" and m.get("tool_calls"):
            parts = []
            if m.get("content"):
                parts.append(genai.protos.Part(text=m["content"]))
            for tc in m["tool_calls"]:
                name = tc["name"] if isinstance(tc, dict) else tc.name
                args = tc["arguments"] if isinstance(tc, dict) else tc.arguments
                parts.append(genai.protos.Part(
                    function_call=genai.protos.FunctionCall(name=name, args=args or {})
                ))
            contents.append(genai.protos.Content(role="model", parts=parts))
        else:
            contents.append(genai.protos.Content(
                role=gemini_role,
                parts=[genai.protos.Part(text=m.get("content", ""))],
            ))
    return contents


def _extract_system(messages: List[Dict[str, Any]]) -> Optional[str]:
    for m in messages:
        if m["role"] == "system":
            return m.get("content")
    return None


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self._genai = genai
        self.model_name = model_name

    def generate(self, messages, *, max_new_tokens=1024, temperature=0.2, stop=None):
        return self._generate(messages, tools=None,
                              max_new_tokens=max_new_tokens, temperature=temperature)

    def generate_with_tools(self, messages, tools, *, max_new_tokens=1024,
                            temperature=0.2, tool_choice="auto"):
        return self._generate(messages, tools=tools,
                              max_new_tokens=max_new_tokens, temperature=temperature)

    def count_tokens(self, text: str) -> int:
        # Approximate — Gemini doesn't expose a tokenizer easily.
        return max(1, len(text) // 4)

    def _generate(self, messages, *, tools, max_new_tokens, temperature):
        system = _extract_system(messages)
        contents = _messages_to_gemini(messages)
        decls = [_to_function_declaration(t) for t in tools] if tools else None

        model = self._genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system,
            tools=decls,
        )
        response = model.generate_content(
            contents,
            generation_config=self._genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_new_tokens,
            ),
        )

        text_parts: List[str] = []
        tool_calls: List[ToolCall] = []
        try:
            for cand in response.candidates:
                for p in cand.content.parts:
                    if getattr(p, "function_call", None) and p.function_call.name:
                        args = {k: v for k, v in p.function_call.args.items()}
                        tool_calls.append(ToolCall(name=p.function_call.name, arguments=args))
                    elif getattr(p, "text", None):
                        text_parts.append(p.text)
        except (AttributeError, ValueError):
            text_parts.append(getattr(response, "text", "") or "")

        usage = getattr(response, "usage_metadata", None)
        return GenerationResult(
            text="".join(text_parts).strip(),
            tool_calls=tool_calls,
            prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            completion_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            finish_reason="tool_call" if tool_calls else "stop",
            raw=response,
        )
