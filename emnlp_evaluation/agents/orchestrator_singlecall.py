"""S-2 ablation: MASDR-RAG with a single synthesis call.

Reviewer asks: "ablation isolating context fragmentation, e.g.,
concatenating all retrieved fragments from multiple tools into a single
synthesis call vs. multi-round interleaving."

This orchestrator behaves exactly like the production MASDR-RAG for the
*planning* and *retrieval* phases — the LLM still emits tool calls and
each tool runs scoped retrieval — but instead of feeding the chunks back
to the LLM in a multi-round tool-observation loop, we collect every
chunk into a single context block and emit one final synthesis call. If
the precision-faithfulness paradox is genuinely caused by context
fragmentation, this variant should preserve MASDR's precision *and*
restore the faithfulness that the multi-round version sheds.
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


MAX_PLAN_HOPS = 1  # single planning call to emit the tool batch


@dataclass
class SingleCallTrace:
    answer: str = ""
    chunks: List[Dict] = field(default_factory=list)
    tool_calls: List[Dict] = field(default_factory=list)
    routed_agents: List[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    wall_time_s: float = 0.0


SYNTH_SYSTEM = (
    "You are a transportation-domain expert. The user query has been "
    "decomposed into sub-queries by a routing planner; the retrieved "
    "passages from every sub-query are provided below in a single context "
    "block. Answer the user query grounded in those passages. If the "
    "passages do not contain the answer, say so explicitly. Cite passages "
    "as [Source N]."
)


def run_singlecall_orchestrator(
    query: str, *,
    llm: LLMProvider, backend: SearchBackend,
) -> SingleCallTrace:
    tools = build_tool_catalog()
    trace = SingleCallTrace()
    t0 = time.time()

    # ---- Plan phase: one LLM call emits one or more tool calls. ----
    plan_messages: List[Dict[str, Any]] = [
        {"role": "system", "content": ORCHESTRATOR_SYSTEM},
        {"role": "user", "content": query},
    ]
    plan = llm.generate_with_tools(plan_messages, tools, max_new_tokens=512)
    trace.llm_calls += 1
    trace.prompt_tokens += plan.prompt_tokens
    trace.completion_tokens += plan.completion_tokens

    if not plan.tool_calls:
        # Planner decided no retrieval needed; fall back to its answer.
        trace.answer = plan.text
        trace.wall_time_s = time.time() - t0
        return trace

    # ---- Retrieval phase: execute every tool call, accumulate chunks. ----
    for tc in plan.tool_calls:
        chunks = _execute_tool(backend, tc.name, tc.arguments)
        trace.chunks.extend(chunks)
        trace.tool_calls.append({"name": tc.name, "arguments": tc.arguments,
                                  "n_chunks": len(chunks)})
        trace.routed_agents.append(TOOL_TO_AGENT.get(tc.name, tc.name))

    # Deduplicate by chunk id, preserving order.
    seen, deduped = set(), []
    for c in trace.chunks:
        if c.get("id") in seen:
            continue
        seen.add(c.get("id"))
        deduped.append(c)
    trace.chunks = deduped

    # ---- Synthesis phase: one call, ALL chunks in a single context. ----
    context = format_chunks(trace.chunks)
    synth_messages = [
        {"role": "system", "content": SYNTH_SYSTEM},
        {"role": "user", "content": f"{query}\n\n--- Retrieved passages ---\n{context}"},
    ]
    synth = llm.generate(synth_messages, max_new_tokens=512)
    trace.llm_calls += 1
    trace.prompt_tokens += synth.prompt_tokens
    trace.completion_tokens += synth.completion_tokens
    trace.answer = synth.text

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
