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
    res = llm.generate(
        [{"role": "user", "content": _ANSWER_PROMPT.format(
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
