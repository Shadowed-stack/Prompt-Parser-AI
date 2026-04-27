"""
PRANAG-AI Prompt-Parser Workflow  (Harshit's LangGraph layer)

State machine:
  parse_node → search_node → research_node → build_node → validate_node
                 ↑                                               |
                 └──────── retry (if validate fails) ───────────┘

Uses LangGraph's StateGraph with a TypedDict state so every node is a pure
function — easy to test, easy to extend.
"""
import uuid
import logging
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END

from shared.config import settings
from orchestrator.prompt_parser import parse_prompt
from orchestrator.research_fetcher import fetch_research
from orchestrator.spec_builder import build_spec
from orchestrator.output_validator import validate_spec
from search_engine.similarity_search import search_traits

logger = logging.getLogger(__name__)


# ── State definition ──────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    prompt:    str
    parsed:    Optional[dict]
    traits:    list
    research:  list
    spec:      Optional[dict]
    validated: Optional[dict]
    error:     Optional[str]
    retries:   int


# ── Node functions ────────────────────────────────────────────────────────────

def parse_node(state: PipelineState) -> PipelineState:
    """LLM: free text → structured ParsedPrompt dict."""
    try:
        parsed = parse_prompt(state["prompt"])
        return {**state, "parsed": parsed, "error": None}
    except Exception as exc:
        logger.error("[parse_node] %s", exc)
        return {**state, "error": str(exc), "retries": state["retries"] + 1}


def search_node(state: PipelineState) -> PipelineState:
    """ChromaDB: retrieve relevant traits for the parsed query."""
    if state.get("error"):
        return state   # skip if upstream already failed

    parsed = state["parsed"] or {}
    query_parts = [
        parsed.get("crop", ""),
        parsed.get("location", ""),
        " ".join(parsed.get("stress_conditions", [])),
        " ".join(parsed.get("target_traits", [])),
    ]
    query = " ".join(p for p in query_parts if p).strip() or state["prompt"]

    try:
        traits = search_traits(query)
        return {**state, "traits": traits}
    except Exception as exc:
        logger.warning("[search_node] %s — continuing with empty traits", exc)
        return {**state, "traits": [], "error": str(exc)}


def research_node(state: PipelineState) -> PipelineState:
    """Semantic Scholar: fetch recent papers matching the query."""
    parsed = state.get("parsed") or {}
    crop   = parsed.get("crop", "crop")
    stress = " ".join(parsed.get("stress_conditions", ["stress"]))
    query  = f"{crop} {stress} tolerance"

    try:
        research = fetch_research(query)
        return {**state, "research": research, "error": None}
    except Exception as exc:
        logger.warning("[research_node] %s — continuing with fallback research", exc)
        return {**state, "research": [], "error": None}   # non-fatal


def build_node(state: PipelineState) -> PipelineState:
    """Assemble all artefacts into a raw spec dict."""
    spec = build_spec(
        state.get("parsed") or {},
        state.get("traits") or [],
        state.get("research") or [],
    )
    return {**state, "spec": spec}


def validate_node(state: PipelineState) -> PipelineState:
    """Pydantic: validate spec dict → serialised Spec or flag retry."""
    validated = validate_spec(state.get("spec") or {})
    if validated:
        return {**state, "validated": validated, "error": None}
    return {
        **state,
        "error": "Spec validation failed.",
        "retries": state["retries"] + 1,
    }


# ── Routing logic ─────────────────────────────────────────────────────────────

def _route_after_validate(state: PipelineState) -> str:
    """Decide whether to retry or finish."""
    if state.get("validated"):
        return "done"
    if state["retries"] < settings.max_retries:
        logger.info(
            "[workflow] Retry %d/%d …", state["retries"], settings.max_retries
        )
        return "retry"
    logger.error("[workflow] Max retries reached.  Returning partial result.")
    return "done"


# ── Graph assembly ─────────────────────────────────────────────────────────────

def _build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("parse",    parse_node)
    graph.add_node("search",   search_node)
    graph.add_node("research", research_node)
    graph.add_node("build",    build_node)
    graph.add_node("validate", validate_node)

    graph.set_entry_point("parse")

    graph.add_edge("parse",    "search")
    graph.add_edge("search",   "research")
    graph.add_edge("research", "build")
    graph.add_edge("build",    "validate")

    graph.add_conditional_edges(
        "validate",
        _route_after_validate,
        {"retry": "parse", "done": END},
    )

    return graph.compile()


_graph = None   # lazy singleton


def run_pipeline(prompt: str) -> dict:
    """
    Entry point for both main.py and app.py.

    Returns:
        {
          "pipeline_id": str,
          "spec":        dict | None,
          "error":       str | None,
        }
    """
    global _graph
    if _graph is None:
        _graph = _build_graph()

    initial_state: PipelineState = {
        "prompt":    prompt,
        "parsed":    None,
        "traits":    [],
        "research":  [],
        "spec":      None,
        "validated": None,
        "error":     None,
        "retries":   0,
    }

    final_state = _graph.invoke(initial_state)

    return {
        "pipeline_id": str(uuid.uuid4()),
        "spec":        final_state.get("validated"),
        "error":       final_state.get("error") if not final_state.get("validated") else None,
    }
