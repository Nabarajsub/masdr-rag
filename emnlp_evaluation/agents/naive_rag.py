"""
Naive RAG baseline: retrieve from the global index, answer. No routing.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List

from ..llm_providers.base import LLMProvider
from .tools_oss import SearchBackend, format_chunks


_ANSWER_PROMPT = """Answer the question using ONLY the retrieved sources. Cite as [Source 1], [Source 2], etc.

Retrieved sources:
{context}

Question: {query}
Answer:"""

# Prompt-control override (final_v3 E5). Monolithic and the scoped systems
# historically shipped different answer prompts (bare here vs a WYDOT-persona
# prompt in hybrid_routed), confounding every monolithic-vs-scoped comparison.
# Setting this makes both use one identical prompt so the comparison isolates
# the retrieval architecture. None = keep original behaviour.
SHARED_ANSWER_PROMPT: str | None = None


def set_shared_answer_prompt(p: str | None) -> None:
    """Force one answer prompt across monolithic and scoped systems, and
    switch the MASDR-RAG orchestrator to its corpus-neutral system prompt, so
    a whole cross-corpus table is free of the answer-prompt confound."""
    global SHARED_ANSWER_PROMPT
    SHARED_ANSWER_PROMPT = p
    from . import hybrid_routed as _hr
    _hr.SHARED_ANSWER_PROMPT = p
    from . import tool_catalog as _tc
    _tc.USE_NEUTRAL_ORCHESTRATOR = p is not None
    from . import react_baseline as _rb
    _rb.USE_NEUTRAL_REACT = p is not None


@dataclass
class NaiveTrace:
    answer: str = ""
    chunks: List[Dict] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    wall_time_s: float = 0.0


def run_naive_rag(query: str, *, llm: LLMProvider, backend: SearchBackend,
                  k: int = 15) -> NaiveTrace:
    trace = NaiveTrace()
    t0 = time.time()
    chunks = backend.global_search(query, limit=k)
    trace.chunks = chunks
    prompt = SHARED_ANSWER_PROMPT or _ANSWER_PROMPT
    res = llm.generate(
        [{"role": "user", "content": prompt.format(
            context=format_chunks(chunks), query=query)}],
        max_new_tokens=1024,
    )
    trace.answer = res.text
    trace.prompt_tokens = res.prompt_tokens
    trace.completion_tokens = res.completion_tokens
    trace.llm_calls = 1
    trace.wall_time_s = time.time() - t0
    return trace


# Alias — Monolithic is the same thing here (single global vector, no scoping).
run_monolithic_rag = run_naive_rag
MonolithicTrace = NaiveTrace
