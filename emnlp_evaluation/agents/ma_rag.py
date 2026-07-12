"""MA-RAG: faithful port of Nguyen et al. 2025 (arXiv 2505.20096)
to our LLMProvider + SearchBackend stack.

Upstream repo: https://github.com/thangylvp/MA-RAG (OpenAI-only).
Prompts are taken verbatim from `src/prompt_template.py` of that repo.
Architecture is a four-agent collaborative chain-of-thought:

    Planner        : Q -> list of sub-tasks
    Step Definer   : (plan, step, memory) -> {type: retrieve|aggregate, task}
    RAG sub-agent  : (sub-query) -> retrieve -> per-doc Extract -> QA answer
    Summary        : (Q, plan, step outputs) -> final answer

Differences vs. upstream that we disclose in the paper:
  * Backbone is Qwen-2.5-7B / Llama-3-8B (vs upstream GPT-4o-mini),
    so structured outputs are obtained by JSON-format prompting +
    regex-tolerant parsing, not OpenAI's `response_format=json_schema`.
  * Retrieval is BGE-M3 against our existing index (matches the
    backbone used by all other systems in the eval -- the comparison
    isolates *agent coordination*, not retriever).
  * MA-RAG operates without organisational metadata: all retrievals
    go through `backend.global_search(query)` (no scope). This is
    the fair contrast to our scoped methods.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..llm_providers.base import LLMProvider
from .tools_oss import SearchBackend, format_chunks


# ── verbatim MA-RAG prompts (src/prompt_template.py @ thangylvp/MA-RAG) ──

PLANNING_SYSTEM = """You are tasked with assisting users in generating structured plans for answering questions. Your goal is to deconstruct a query into manageable, simpler components. For each question, perform the following tasks:

*Analysis: Identify the core components of the question, emphasizing the key elements and context needed for a comprehensive understanding. Determine whether the question is straightforward or requires multiple steps to provide an accurate answer.

*Plan Creation:
- Break down the question into smaller, simpler questions by reasoning that lead to the final answer. Ensure those steps are non overlap. Stop at the step where its answer can be the final answer.
- Ensure each step is clear and logically sequenced.
- Consider any past attempts or experiences provided as context, and use them to refine or adjust the plan to avoid past pitfalls.
- Each step is a question to search, or to aggregate output from previous steps. Do not verify previous step.
- Your task is planning, not answering. Do not put any answer from your knowledge into the plan.

# Notes:
- Your task is to provide clarity and guidance on the approach to answering, rather than providing the final answer directly.
- Put your output in a list of string, each string describe a sub-task

# Example plan:
Question: What country of origin does House of Cosbys and Bill Cosby have in common?
Steps: ["Determine the country of origin for House of Cosbys.", "Determine the country of origin for Bill Cosby.", "From previous answers, which is the common country"]
Question: Which film has the director who died later, The House Of Tears or College Ranga?
Steps: ["Identify the director of The House Of Tears", "Identify the director of College Ranga", "When did the director of The House Of Tears die", "When did the director of College Ranga die", "Compare the death dates of the two directors to determine which one died later."]
Question: Peter Griffith's granddaughter had her screen debut in what 1999 film?
Steps: ["Who is Peter Griffith's granddaughter", "What 1999 film did she have screen debut"]
Question: how many episodes are in chicago fire season 4?
Steps: ["how many episodes are in chicago fire season 4"]

Return ONLY valid JSON of the form:
{"analysis": "<your analysis>", "step": ["sub-task 1", "sub-task 2", ...]}
"""

PLANNING_HUMAN = """Question: {question}?
Past experience:
{memory}
"""

STEP_SYSTEM = """
Given a plan, the current step, and the results from finished steps, decide the task for this step.
Output the type of task and the query.
The query need to be in detail (do not put "based on the previous results" in the query)
Include all of information from previous step's results in the query if it maked, especially for aggregate task
Be concise.

Return ONLY valid JSON of the form:
{"type": "aggregate" or "question-answering", "task": "<the detailed task / query>"}
"""

STEP_HUMAN = """Plan: {plan}
Current step: {cur_step}
Results of finished steps:
{memory}
"""

EXTRACT_SYSTEM = """Summarize and extract all relevant information from the provided passages based on the given question. Remove all irrelevant information. Think step-by-step.

# Steps

1. **Identify Key Elements**: Read the question carefully to determine what specific information is being requested.
2. **Analyze Passages**: Review the passages thoroughly to find any segments that contain information relevant to the question.
3. **Extract Relevant Information**: Highlight or note down sentences, phrases, or words from the passages that relate to the question.
4. **Remove Irrelevant Details**: Ensure that all extracted information is relevant to the question, eliminating any unnecessary or unrelated content.

# Output Format
- Output a list of notes. Each note contains related information from the passage as well as precise evidences and why.
- Each note is clear, standalone.

# Notes
- Avoiding any irrelevant details.
- If a piece of information is mentioned in multiple places, include it only once.
- If there are no related information, output: No related information from this document."""

EXTRACT_HUMAN = """
Passage:
###
{passage}
###

Query: {question}?
"""

QA_SYSTEM = """You are an assistant for question-answering tasks. Use the following process to deliver concise and precise answers based on the retrieved context. If all of retrieved context are not relevant, answer based on general knowledge.

1. **Analyze Carefully**: Begin by thoroughly analyzing both the question and the provided context.

2. **Identify Core Details**: Focus on identifying the essential names, terms, or details that directly answer the question. Disregard any irrelevant information.

3. **Provide a Concise Answer**:
   - Remove redundant words and extraneous details.
   - Present the answer by listing only the necessary names, terms, or very brief facts that are crucial for answering the question.

4. **Clarity and Accuracy**: Ensure that your answer is clear and maintains the original meaning of the information provided.

5. **consensus**: If the contexts are not consensus, pick one which is the most logical, consensus, or confident.

6. **IMPORTANT**: If the provided context couldn't bring any related information, answer by your self.

Return ONLY valid JSON of the form:
{"analysis": "<your reasoning>", "answer": "<concise answer>", "success": "Yes" or "No", "rating": <0-10 integer>}
"""

QA_HUMAN = """
Retrieved documents:
{context}
Question: {question}
"""

AGGREGATE_SYSTEM = """Answer the question from human.
Provide a Concise Answer:
- Remove redundant words and extraneous details.
- Present the answer by listing only the necessary names, terms, or very brief facts that are crucial for answering the question.
- If you have multiple answers, only output one answer which is most confident
Think step-by-step

Return ONLY valid JSON of the form:
{"analysis": "<your reasoning>", "answer": "<concise answer>", "success": "Yes" or "No", "rating": <0-10 integer>}
"""

AGGREGATE_HUMAN = """{question}"""

SUMMARY_SYSTEM = """Your task is writing a summary about a plan to solve a question and answer question based on outputs from each step in the plan. Think step-by-step.

** Input
- The original question
- The plan: a sequence of sub-task.
- Output of each step in the plan.

** Output
- If all of steps are solved, output the final answer for the original question by combining step's output.
- If one or many of steps are unsolved, but you can still find the answer based on step's output, output the final answer.
- If you could not find the final answer for the question, output Unsuccessful.

Return ONLY valid JSON of the form:
{"output": "Successful" or "Unsuccessful: <reason>", "answer": "<final answer>", "score": <0-10 integer>}
"""

SUMMARY_HUMAN = """Original Question: {question}
Plan: {plan}
Output of steps:
{memory}

Original Question: {question}
"""


# ── JSON-output parsing (tolerant of trailing prose / code fences) ──

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _parse_json(text: str, default: dict) -> dict:
    """Extract the first JSON object from text, tolerating code fences."""
    if not text:
        return default
    m = _CODE_FENCE_RE.search(text)
    candidate = m.group(1) if m else None
    if candidate is None:
        m = _JSON_BLOCK_RE.search(text)
        candidate = m.group(0) if m else None
    if candidate is None:
        return default
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        try:
            return json.loads(candidate.replace("'", '"'))
        except json.JSONDecodeError:
            return default


def _as_text(v) -> str:
    """Coerce an LLM-produced answer field to a string. The model
    sometimes emits a JSON array or object for "answer"; flatten it so
    trace.answer is always a string for the downstream judge."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return " ".join(_as_text(x) for x in v)
    if isinstance(v, dict):
        return " ".join(_as_text(x) for x in v.values())
    return str(v)


def _answer_or_raw(parsed: dict, raw: str) -> str:
    """Return the parsed ``answer`` field, or fall back to the raw model
    text only when JSON parsing failed entirely. If the model emitted
    valid JSON whose ``answer`` happens to be empty (an honest "I
    can't answer"), return ``""`` so the caller's fallback chain
    substitutes prose from step_outputs/analysis. Returning ``raw``
    in that case would leak the JSON-shaped blob itself
    (e.g. ``{"output": "...", "answer": "", "score": 0}``) as the
    final answer."""
    if parsed:
        # JSON parsed; trust its answer field (possibly empty).
        return _as_text(parsed.get("answer", "")).strip()
    raw_stripped = raw.strip()
    # _parse_json failed: raw may contain prose, or may be a malformed
    # JSON blob that the regex parser couldn't pin down. In the latter
    # case the raw text is useless as an answer.
    if raw_stripped.startswith("{") and '"answer"' in raw_stripped:
        return ""
    return raw_stripped


# ── trace ──

@dataclass
class MaRagTrace:
    answer: str = ""
    chunks: List[Dict] = field(default_factory=list)
    plan: List[str] = field(default_factory=list)
    step_outputs: List[Dict] = field(default_factory=list)
    step_notes: List[List[str]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    wall_time_s: float = 0.0
    early_stop_reason: str = "summary"


# ── individual agent calls ──

def _gen(llm: LLMProvider, sys: str, usr: str, *, trace: MaRagTrace,
         max_new_tokens: int = 768, temperature: float = 0.2) -> str:
    res = llm.generate(
        [{"role": "system", "content": sys},
         {"role": "user", "content": usr}],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    trace.llm_calls += 1
    trace.prompt_tokens += res.prompt_tokens
    trace.completion_tokens += res.completion_tokens
    return res.text


def _planner(llm: LLMProvider, question: str, *, trace: MaRagTrace) -> List[str]:
    out = _gen(llm,
               PLANNING_SYSTEM,
               PLANNING_HUMAN.format(question=question, memory="empty"),
               trace=trace, max_new_tokens=512, temperature=0.3)
    parsed = _parse_json(out, default={"analysis": "", "step": [question]})
    steps = parsed.get("step") or [question]
    if isinstance(steps, str):  # tolerate a single-string degenerate output
        steps = [steps]
    return [str(s) for s in steps][:6]   # safety cap


def _step_definer(llm: LLMProvider, plan: List[str], step_idx: int,
                  step_outputs: List[Dict], *, trace: MaRagTrace) -> Dict:
    plan_s = "[" + ", ".join(plan) + "]"
    cur_step = plan[step_idx]
    memory = ""
    for j, (s, out) in enumerate(zip(plan[:step_idx], step_outputs)):
        memory += f"Task: {s}\nAnswer: {out.get('answer','')}\n\n"
    out = _gen(llm,
               STEP_SYSTEM,
               STEP_HUMAN.format(plan=plan_s, cur_step=cur_step, memory=memory or "(none)"),
               trace=trace, max_new_tokens=256, temperature=0.3)
    parsed = _parse_json(out, default={"type": "question-answering", "task": cur_step})
    return {"type": parsed.get("type", "question-answering"),
            "task": parsed.get("task", cur_step)}


def _rag_subagent(llm: LLMProvider, backend: SearchBackend, sub_query: str,
                  *, trace: MaRagTrace, top_k: int = 10) -> Dict:
    """Retrieve → per-doc extract → QA."""
    chunks = backend.global_search(sub_query)[:top_k]
    trace.chunks.extend(chunks)

    notes = []
    for c in chunks:
        passage = c.get("text", "")
        if not passage:
            continue
        out = _gen(llm,
                   EXTRACT_SYSTEM,
                   EXTRACT_HUMAN.format(passage=passage, question=sub_query),
                   trace=trace, max_new_tokens=512, temperature=0.0)
        notes.append(f"[{out.strip()}]")
    trace.step_notes.append(notes)

    context_parts = []
    for c, note in zip(chunks, notes):
        cid = c.get("id", "?")
        context_parts.append(f"doc_{cid}: {note}")
    context = "\n\n".join(context_parts) if context_parts else "(no documents)"
    qa_out = _gen(llm,
                  QA_SYSTEM,
                  QA_HUMAN.format(context=context, question=sub_query),
                  trace=trace, max_new_tokens=512, temperature=0.3)
    parsed = _parse_json(qa_out, default={})
    return {
        "analysis": parsed.get("analysis", ""),
        "answer": _answer_or_raw(parsed, qa_out),
        "success": parsed.get("success", "No"),
        "rating": parsed.get("rating", 0),
        "type": "question-answering",
    }


def _aggregate(llm: LLMProvider, aggregate_query: str, *, trace: MaRagTrace) -> Dict:
    out = _gen(llm,
               AGGREGATE_SYSTEM,
               AGGREGATE_HUMAN.format(question=aggregate_query),
               trace=trace, max_new_tokens=512, temperature=0.0)
    parsed = _parse_json(out, default={})
    return {
        "analysis": parsed.get("analysis", ""),
        "answer": _answer_or_raw(parsed, out),
        "success": parsed.get("success", "No"),
        "rating": parsed.get("rating", 0),
        "type": "aggregate",
    }


def _summarise(llm: LLMProvider, question: str, plan: List[str],
               step_outputs: List[Dict], *, trace: MaRagTrace) -> Dict:
    plan_s = "[" + ", ".join(plan) + "]"
    memory = ""
    for s, out in zip(plan, step_outputs):
        memory += (f"Task: {s}\n"
                   f"Answer: {out.get('answer','')}\n"
                   f"Confident score: {out.get('rating',0)}\n\n")
    out = _gen(llm,
               SUMMARY_SYSTEM,
               SUMMARY_HUMAN.format(question=question, plan=plan_s, memory=memory),
               trace=trace, max_new_tokens=512, temperature=0.0)
    parsed = _parse_json(out, default={})
    return {
        "output": parsed.get("output", "Unsuccessful"),
        "answer": _answer_or_raw(parsed, out),
        "score": parsed.get("score", 0),
    }


# ── public entry point ──

def run_ma_rag(query: str, *, llm: LLMProvider, backend: SearchBackend,
               max_steps: int = 6, top_k: int = 10) -> MaRagTrace:
    """MA-RAG end-to-end: plan → execute steps → summarise.

    Apples-to-apples with our other systems on the same LLM + retriever.
    The only difference from MASDR-RAG is the *coordination protocol*:
    sequential plan-decompose-execute with per-doc extraction, vs scoped
    multi-agent retrieval with one synthesis call.
    """
    trace = MaRagTrace()
    t0 = time.time()

    # 1) Plan
    plan = _planner(llm, query, trace=trace)
    trace.plan = plan

    if not plan:
        trace.early_stop_reason = "empty_plan"
        trace.wall_time_s = time.time() - t0
        return trace

    # 2) Execute steps
    step_outputs: List[Dict] = []
    for i in range(min(len(plan), max_steps)):
        # last step success==No short-circuits per upstream
        if i > 0 and step_outputs[-1].get("success", "").lower() == "no":
            trace.early_stop_reason = "step_failed"
            break

        defn = _step_definer(llm, plan, i, step_outputs, trace=trace)
        if defn["type"].lower().startswith("agg"):
            out = _aggregate(llm, defn["task"], trace=trace)
        else:
            out = _rag_subagent(llm, backend, defn["task"], trace=trace, top_k=top_k)
        step_outputs.append(out)
    trace.step_outputs = step_outputs

    # 3) Summarise
    summary = _summarise(llm, query, plan, step_outputs, trace=trace)
    trace.answer = _as_text(summary.get("answer")) or (
        _as_text(step_outputs[-1].get("answer", "")) if step_outputs else ""
    )
    # Final safety net: never emit an empty answer. Fall back to the
    # most recent non-empty step answer, then to any step's analysis.
    if not trace.answer.strip():
        for so in reversed(step_outputs):
            cand = (_as_text(so.get("answer", "")).strip()
                    or _as_text(so.get("analysis", "")).strip())
            if cand:
                trace.answer = cand
                break

    # dedupe chunks by id
    seen, dedup = set(), []
    for c in trace.chunks:
        cid = c.get("id")
        if cid in seen:
            continue
        seen.add(cid); dedup.append(c)
    trace.chunks = dedup

    trace.wall_time_s = time.time() - t0
    return trace
