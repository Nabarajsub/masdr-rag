"""
MASDR-RAG (multi-agent, orchestrator-routed) — provider-agnostic.

This is the open-source counterpart of `agentic_solution/orchestrator.py`.
The LLM is supplied as an `LLMProvider`; the retrieval backend as a
`SearchBackend`. Everything else mirrors the production behaviour.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..llm_providers.base import LLMProvider
from .tool_catalog import (
    ORCHESTRATOR_SYSTEM,
    TOOL_TO_AGENT,
    build_tool_catalog,
)
from .tools_oss import SearchBackend, format_chunks


MAX_TOOL_HOPS = 5


@dataclass
class RunTrace:
    answer: str = ""
    chunks: List[Dict] = field(default_factory=list)         # union across hops
    chunks_per_hop: List[List[Dict]] = field(default_factory=list)
    tool_calls: List[Dict] = field(default_factory=list)
    routed_agents: List[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    wall_time_s: float = 0.0


def run_orchestrator(
    query: str, *,
    llm: LLMProvider, backend: SearchBackend,
    max_hops: int = MAX_TOOL_HOPS,
) -> RunTrace:
    tools = build_tool_catalog()
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": ORCHESTRATOR_SYSTEM},
        {"role": "user", "content": query},
    ]
    trace = RunTrace()
    t0 = time.time()

    for _ in range(max_hops):
        result = llm.generate_with_tools(messages, tools, max_new_tokens=1024)
        trace.llm_calls += 1
        trace.prompt_tokens += result.prompt_tokens
        trace.completion_tokens += result.completion_tokens

        if not result.tool_calls:
            trace.answer = result.text
            break

        # Append assistant turn with the function calls
        messages.append({
            "role": "assistant",
            "content": result.text,
            "tool_calls": [
                {"name": tc.name, "arguments": tc.arguments} for tc in result.tool_calls
            ],
        })

        # Execute every tool call in sequence, append responses
        hop_chunks: List[Dict] = []
        for tc in result.tool_calls:
            chunks = _execute_tool(backend, tc.name, tc.arguments)
            hop_chunks.extend(chunks)
            trace.tool_calls.append({"name": tc.name, "arguments": tc.arguments,
                                      "n_chunks": len(chunks)})
            trace.routed_agents.append(TOOL_TO_AGENT.get(tc.name, tc.name))
            messages.append({
                "role": "tool",
                "name": tc.name,
                "content": format_chunks(chunks),
            })
        trace.chunks_per_hop.append(hop_chunks)
        trace.chunks.extend(hop_chunks)

    # Deduplicate chunk union by id
    seen, deduped = set(), []
    for c in trace.chunks:
        if c.get("id") in seen:
            continue
        seen.add(c.get("id"))
        deduped.append(c)
    trace.chunks = deduped

    trace.wall_time_s = time.time() - t0
    return trace


def _execute_tool(backend: SearchBackend, tool_name: str, args: Dict[str, Any]) -> List[Dict]:
    agent_name = TOOL_TO_AGENT.get(tool_name)
    if not agent_name:
        return []
    q = args.get("query", "")
    year = args.get("year")
    section = args.get("section")
    if agent_name == "general_agent":
        return backend.global_search(q, year=year)
    return backend.combined_scoped_search(q, agent_name, year=year, section=section)
