"""
LangChain `AgentExecutor` baseline.

We use LangChain's standard ReAct agent (`create_react_agent` / `AgentExecutor`)
over the same nine scoped retrieval tools the orchestrator and our hand-rolled
ReAct use. The LLM is wrapped as a custom LangChain LLM that calls our
LLMProvider, so the comparison is apples-to-apples: same model, same tools,
same chunks — only the agent framework differs.

This is the "third-party multi-agent framework" baseline EMNLP reviewers
typically ask for ("is the overhead from your custom orchestrator or from
the iterative-reasoning paradigm?").
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import PromptTemplate

from ..llm_providers.base import LLMProvider
from .tool_catalog import TOOL_TO_AGENT
from .tools_oss import SearchBackend, format_chunks


# Standard hwchase17/react-style template, minus the chat-style scaffolding.
_REACT_PROMPT = """Answer the following question as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action (a JSON object like {{"query": "..."}})
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat at most 6 times)
Thought: I now know the final answer
Final Answer: the final answer to the original question, with [Source N] citations.

Begin!

Question: {input}
Thought:{agent_scratchpad}"""


class QwenLangChainLLM(LLM):
    """Adapter so LangChain can call our LLMProvider as if it were an OpenAI text LLM."""

    provider: Any  # LLMProvider, but pydantic v2 hates ABCs here
    last_prompt_tokens: int = 0
    last_completion_tokens: int = 0
    last_n_calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "qwen-emnlp-eval"

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[Any] = None, **kwargs: Any) -> str:
        res = self.provider.generate(
            [{"role": "user", "content": prompt}],
            max_new_tokens=512, temperature=0.2,
        )
        self.last_prompt_tokens += res.prompt_tokens
        self.last_completion_tokens += res.completion_tokens
        self.last_n_calls += 1
        return res.text


def _make_tool_fn(backend: SearchBackend, agent_name: str):
    """Closure that captures agent_name correctly (avoids late-binding pitfalls)."""
    import json as _json
    def _fn(query: str) -> str:
        try:
            args = _json.loads(query)
            q = args.get("query", query)
        except Exception:
            q = query
        if agent_name == "general_agent":
            chunks = backend.global_search(q)
        else:
            chunks = backend.combined_scoped_search(q, agent_name)
        return format_chunks(chunks[:10])
    return _fn


def _make_tools(backend: SearchBackend) -> List[Tool]:
    """One LangChain Tool per scoped agent + a general fallback."""
    tools: List[Tool] = []
    for tool_name, agent_name in TOOL_TO_AGENT.items():
        tools.append(Tool(
            name=tool_name,
            description=f"Search the {agent_name.replace('_',' ')} document collection.",
            func=_make_tool_fn(backend, agent_name),
        ))
    return tools


@dataclass
class LangChainTrace:
    answer: str = ""
    chunks: List[Dict] = field(default_factory=list)
    iterations: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    wall_time_s: float = 0.0
    early_stop_reason: str = "final_answer"


def run_langchain(query: str, *, llm: LLMProvider, backend: SearchBackend,
                  max_iters: int = 6) -> LangChainTrace:
    """Run a single query through LangChain's AgentExecutor."""
    trace = LangChainTrace()
    t0 = time.time()

    wrapper = QwenLangChainLLM(provider=llm)
    tools = _make_tools(backend)
    prompt = PromptTemplate.from_template(_REACT_PROMPT)
    agent = create_react_agent(wrapper, tools, prompt)
    executor = AgentExecutor(
        agent=agent, tools=tools,
        max_iterations=max_iters,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )
    try:
        out = executor.invoke({"input": query})
    except Exception as e:
        trace.answer = f"[langchain-error] {type(e).__name__}: {e}"
        trace.wall_time_s = time.time() - t0
        return trace

    trace.answer = out.get("output", "") or ""
    steps = out.get("intermediate_steps", [])
    trace.iterations = len(steps)
    trace.early_stop_reason = "max_iters" if trace.iterations >= max_iters else "final_answer"
    trace.prompt_tokens = wrapper.last_prompt_tokens
    trace.completion_tokens = wrapper.last_completion_tokens
    trace.llm_calls = wrapper.last_n_calls
    trace.wall_time_s = time.time() - t0
    return trace
