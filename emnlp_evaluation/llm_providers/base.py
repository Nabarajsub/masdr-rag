"""
Provider-agnostic LLM interface.

Every backend (Gemini, Qwen, Llama, ...) returns a `GenerationResult`. The
runner code never imports a vendor SDK directly — it asks the factory for a
provider and calls `.generate(...)` or `.generate_with_tools(...)`.

Tool schemas use the OpenAI/JSON-Schema shape because both Qwen2.5 and Llama-3
chat templates accept that format natively. The Gemini adapter translates to
`genai.protos.FunctionDeclaration` internally.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class ToolSpec:
    """OpenAI/JSON-Schema-shaped tool description."""
    name: str
    description: str
    parameters: Dict[str, Any]   # JSON schema object

    def to_openai(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]
    call_id: Optional[str] = None


@dataclass
class GenerationResult:
    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"
    raw: Any = None      # provider-specific response, for debugging only

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMProvider(Protocol):
    """Minimal contract every backend must implement."""

    name: str

    def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        max_new_tokens: int = 1024,
        temperature: float = 0.2,
        stop: Optional[List[str]] = None,
    ) -> GenerationResult: ...

    def generate_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[ToolSpec],
        *,
        max_new_tokens: int = 1024,
        temperature: float = 0.2,
        tool_choice: str = "auto",
    ) -> GenerationResult: ...

    def count_tokens(self, text: str) -> int: ...
