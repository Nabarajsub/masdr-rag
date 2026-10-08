"""
Tool catalog shared by orchestrator / hybrid / ReAct.

Each entry describes one of the nine scoped domain searches as an
OpenAI-style JSON-Schema tool spec, paired with the agent name that backs it.
"""
from __future__ import annotations

from typing import Dict, List

from ..llm_providers.base import ToolSpec


# tool_name → agent_name in AGENT_SERIES_FILTERS
TOOL_TO_AGENT: Dict[str, str] = {
    "search_specs":              "specs_agent",
    "search_construction_manual": "construction_agent",
    "search_materials_testing":  "materials_agent",
    "search_design_manual":      "design_agent",
    "search_crash_data":         "safety_agent",
    "search_bridge_program":     "bridge_agent",
    "search_stip_planning":      "planning_agent",
    "search_admin_reports":      "admin_agent",
    "search_general":            "general_agent",
}


_TOOL_DESCRIPTIONS = {
    "search_specs": (
        "Search Wyoming Standard Specifications for construction requirements, "
        "material specs, tolerances, thresholds, contractor obligations, penalties, "
        "warranties, insurance, change orders, certifications, work zone safety, "
        "concrete/asphalt/aggregate, guardrails, culverts, welding, pile driving, "
        "or any 'Section XXX' reference."
    ),
    "search_construction_manual": (
        "Search Construction Manuals for field inspection procedures, project "
        "administration, and construction management processes."
    ),
    "search_materials_testing": (
        "Search Materials Testing Manuals for lab test procedures, sampling "
        "methods, testing frequencies, and material acceptance criteria."
    ),
    "search_design_manual": (
        "Search Design Manuals for road/bridge geometric design, alignment, "
        "cross sections, and CADD standards."
    ),
    "search_crash_data": (
        "Search Traffic Crash Reports and Highway Safety Plans for statistics, "
        "fatalities by county/year, accident trends, and impaired driving data."
    ),
    "search_bridge_program": (
        "Search Bridge Program documents for bridge design, load ratings, "
        "bridge plans, and structural details."
    ),
    "search_stip_planning": (
        "Search STIP and Corridor Studies for project funding, planned projects, "
        "and transportation improvement programming."
    ),
    "search_admin_reports": (
        "Search WYDOT Annual Reports, Financial Reports, and Operating Budgets "
        "for organizational, financial, and leadership info."
    ),
    "search_general": (
        "Search ALL documents. Use when the query does not fit a specific domain "
        "(e.g., driver licenses, vehicle registration, general WYDOT info)."
    ),
}


_PARAMS_SEARCH = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Search query."},
        "year": {"type": "integer", "description": "Optional year filter (e.g. 2021)."},
        "section": {"type": "string", "description": "Optional section number filter (e.g. '414')."},
    },
    "required": ["query"],
}


def build_tool_catalog() -> List[ToolSpec]:
    return [
        ToolSpec(name=name, description=_TOOL_DESCRIPTIONS[name], parameters=_PARAMS_SEARCH)
        for name in TOOL_TO_AGENT
    ]


ORCHESTRATOR_SYSTEM = """You are the WYDOT Knowledge Graph Assistant — a helpful AI that answers questions about Wyoming Department of Transportation using a knowledge graph of WYDOT documents.

You MUST call at least one search tool for every query. If the query is general or cross-domain, call `search_general`.

For each user query:
1. Decide which tool(s) to call.
2. Call them with a clear search query.
3. Read the returned chunks carefully.
4. Synthesize a concise answer with citations.

CITATION RULES:
- Reference sources as [Source 1], [Source 2], etc. matching the source number in the returned chunks.
- Include document title, section, and year when citing.

MULTI-STEP:
- For comparison queries, call the same tool twice with different `year` values.
- For cross-domain queries, call multiple tools.
- At most 5 tool calls per query.

Be thorough but concise. Always ground answers in the retrieved content."""

# Corpus-neutral variant (final_v3 E5 prompt control) — same tool-calling
# instructions, WYDOT persona removed, so the orchestrator's system prompt is
# not a corpus-mismatched confound on non-WYDOT benchmarks.
ORCHESTRATOR_SYSTEM_NEUTRAL = """You are a helpful AI assistant that answers questions using a searchable collection of documents.

You MUST call at least one search tool for every query. If the query is general or cross-domain, call `search_general`.

For each user query:
1. Decide which tool(s) to call.
2. Call them with a clear search query.
3. Read the returned chunks carefully.
4. Synthesize a concise answer with citations.

CITATION RULES:
- Reference sources as [Source 1], [Source 2], etc. matching the source number in the returned chunks.
- Include document title, section, and year when citing.

MULTI-STEP:
- For comparison queries, call the same tool twice with different `year` values.
- For cross-domain queries, call multiple tools.
- At most 5 tool calls per query.

Be thorough but concise. Always ground answers in the retrieved content."""
