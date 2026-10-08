"""
ReAct baseline.

The agent emits Thought / Action / Observation cycles. Action lines must be
of the form `Action: tool_name({"query": "...", ...})`. The same nine scoped
search tools as the orchestrator are exposed, so the comparison is apples-to-
apples in terms of retrieval capability. The difference is the control loop:
ReAct decides ONE action per turn and re-prompts the LLM after observing the
result, vs. the orchestrator which can batch tool calls.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List

from ..llm_providers.base import LLMProvider
from .tool_catalog import TOOL_TO_AGENT
from .tools_oss import SearchBackend, format_chunks


# Persona-neutral variant (shared-prompt control): identical loop protocol and
# rules, corpus-specific persona removed, so ReAct is not penalised by a
# corpus-mismatched persona on non-WYDOT corpora. Selected when
# USE_NEUTRAL_REACT is set by naive_rag.set_shared_answer_prompt().
USE_NEUTRAL_REACT = False

REACT_SYSTEM_NEUTRAL = """You are a research agent answering questions using a searchable collection of documents.

You operate in a loop of:
  Thought: <what you need next>
  Action: <tool_name>({{"query": "...", "year": 2021}})
  Observation: <tool output, supplied by the user>

When you have enough information, end with:
  Final Answer: <your answer with [Source N] citations>

Available tools:
{tool_list}

Rules:
- Emit EXACTLY ONE Action per turn (or Final Answer).
- Do NOT invent observations --- wait for them to be provided.
- After at most 6 actions, emit Final Answer even if uncertain.
- Cite chunks as [Source 1], [Source 2], etc. matching the chunk numbering in observations.
"""


REACT_SYSTEM = """You are a research agent answering questions about Wyoming Department of Transportation documents.

You operate in a loop of:
  Thought: <what you need next>
  Action: <tool_name>({{"query": "...", "year": 2021}})
  Observation: <tool output, supplied by the user>

When you have enough information, end with:
  Final Answer: <your answer with [Source N] citations>

Available tools:
{tool_list}

Rules:
- Emit EXACTLY ONE Action per turn (or Final Answer).
- Do NOT invent observations — wait for them to be provided.
- After at most 6 actions, emit Final Answer even if uncertain.
- Cite chunks as [Source 1], [Source 2], etc. matching the chunk numbering in observations.
"""


TOOL_DESCRIPTIONS = {
    "search_specs": "Search Wyoming Standard Specifications.",
    "search_construction_manual": "Search Construction Manuals.",
    "search_materials_testing": "Search Materials Testing Manuals.",
    "search_design_manual": "Search Design Manuals.",
    "search_crash_data": "Search Traffic Crash Reports.",
    "search_bridge_program": "Search Bridge Program documents.",
    "search_stip_planning": "Search STIP and corridor studies.",
    "search_admin_reports": "Search Annual Reports and budgets.",
    "search_general": "Search ALL documents (fallback).",
}


_ACTION_RE = re.compile(r"Action:\s*([a-zA-Z_]+)\s*\(\s*(\{.*?\})\s*\)", re.DOTALL)
_FINAL_RE = re.compile(r"Final\s+Answer:\s*(.*)", re.DOTALL | re.IGNORECASE)


@dataclass
class ReActTrace:
    answer: str = ""
    chunks: List[Dict] = field(default_factory=list)
    iterations: int = 0
    actions: List[Dict] = field(default_factory=list)
    routed_agents: List[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    wall_time_s: float = 0.0
    early_stop_reason: str = "final_answer"


def _system_prompt() -> str:
    tool_list = "\n".join(f"  - {k}: {v}" for k, v in TOOL_DESCRIPTIONS.items())
    base = REACT_SYSTEM_NEUTRAL if USE_NEUTRAL_REACT else REACT_SYSTEM
    return base.format(tool_list=tool_list)


def _parse_action(text: str):
    m = _ACTION_RE.search(text)
    if not m:
        return None, None
    name = m.group(1).strip()
    try:
        args = json.loads(m.group(2))
    except json.JSONDecodeError:
        # try replacing single quotes
        try:
            args = json.loads(m.group(2).replace("'", '"'))
        except json.JSONDecodeError:
            args = {"query": m.group(2)}
    return name, args


def _exec_action(backend: SearchBackend, name: str, args: dict) -> List[Dict]:
    agent = TOOL_TO_AGENT.get(name)
    if not agent:
        return []
    q = args.get("query", "")
    year = args.get("year")
    section = args.get("section")
    if agent == "general_agent":
        return backend.global_search(q, year=year)
    return backend.combined_scoped_search(q, agent, year=year, section=section)


def run_react(query: str, *, llm: LLMProvider, backend: SearchBackend,
              max_iters: int = 6) -> ReActTrace:
    trace = ReActTrace()
    t0 = time.time()

    messages = [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": f"Question: {query}\n\nBegin."},
    ]

    for i in range(max_iters):
        res = llm.generate(messages, max_new_tokens=512)
        trace.llm_calls += 1
        trace.prompt_tokens += res.prompt_tokens
        trace.completion_tokens += res.completion_tokens
        trace.iterations = i + 1

        text = res.text
        messages.append({"role": "assistant", "content": text})

        final = _FINAL_RE.search(text)
        if final:
            trace.answer = final.group(1).strip()
            trace.early_stop_reason = "final_answer"
            break

        name, args = _parse_action(text)
        if not name:
            # No action and no final answer — nudge once, then bail.
            messages.append({
                "role": "user",
                "content": "You must emit either an Action: tool(...) or a Final Answer:.",
            })
            continue

        chunks = _exec_action(backend, name, args or {})
        trace.actions.append({"name": name, "arguments": args, "n_chunks": len(chunks)})
        trace.routed_agents.append(TOOL_TO_AGENT.get(name, name))
        trace.chunks.extend(chunks)

        obs = format_chunks(chunks)
        messages.append({"role": "user", "content": f"Observation:\n{obs}"})
    else:
        trace.early_stop_reason = "max_iters"
        if not trace.answer:
            # ask once for a final answer using what we've got
            messages.append({"role": "user", "content": "Stop. Emit Final Answer: now."})
            res = llm.generate(messages, max_new_tokens=512)
            trace.llm_calls += 1
            trace.prompt_tokens += res.prompt_tokens
            trace.completion_tokens += res.completion_tokens
            m = _FINAL_RE.search(res.text)
            trace.answer = m.group(1).strip() if m else res.text.strip()

    # dedupe chunks by id
    seen, dedup = set(), []
    for c in trace.chunks:
        if c.get("id") in seen:
            continue
        seen.add(c.get("id")); dedup.append(c)
    trace.chunks = dedup

    trace.wall_time_s = time.time() - t0
    return trace
