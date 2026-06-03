"""
PRANAG-AI Prompt-Parser Workflow  (LangGraph StateGraph)

Fix over v1:
  • search_node no longer bails on parse error — uses original prompt as query
    so retrieved_traits are always populated even when LLM fails.
  • research_node always runs (non-fatal).
  • Better logging per node.
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
from orchestrator.output_exporter import export_spec        # NEW
from search_engine.similarity_search import search_traits

logger = logging.getLogger(__name__)


# ── State ─────────────────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    prompt:    str
    pipeline_id: str          # carry the ID through all nodes
    parsed:    Optional[dict]
    traits:    list
    research:  list
    spec:      Optional[dict]
    validated: Optional[dict]
    error:     Optional[str]
    retries:   int


# ── Nodes ─────────────────────────────────────────────────────────────────────

def parse_node(state: PipelineState) -> PipelineState:
    """LLM + regex: free text → ParsedPrompt dict.
    parse_prompt() now never fully fails — regex fallback guarantees a result."""
    try:
        parsed = parse_prompt(state["prompt"])
        return {**state, "parsed": parsed, "error": None}
    except Exception as exc:
        logger.error("[parse_node] Unexpected error: %s", exc)
        return {**state, "parsed": None, "error": str(exc), "retries": state["retries"] + 1}


def search_node(state: PipelineState) -> PipelineState:
    """ChromaDB: retrieve relevant traits.

    KEY FIX: we no longer skip when there's a parse error.
    Even if parsed is None, we fall back to searching with the raw prompt.
    """
    parsed = state.get("parsed") or {}

    # Build a rich query from whatever we know
    query_parts = [
        parsed.get("crop", ""),
        parsed.get("location", ""),
        " ".join(parsed.get("stress_conditions", [])),
        " ".join(parsed.get("target_traits", [])),
    ]
    query = " ".join(p for p in query_parts if p).strip()

    # Last resort: use the original prompt directly
    if not query or query.strip() == "":
        query = state["prompt"]

    try:
        crop = parsed.get("crop") or None
        traits = search_traits(query, crop=crop)
        logger.info("[search_node] Retrieved %d traits for query: '%s'", len(traits), query[:60])
        return {**state, "traits": traits}
    except Exception as exc:
        logger.warning("[search_node] Search failed: %s", exc)
        return {**state, "traits": []}


def research_node(state: PipelineState) -> PipelineState:
    """Semantic Scholar: fetch relevant papers."""
    parsed   = state.get("parsed") or {}
    crop     = parsed.get("crop") or "crop"
    stress   = " ".join(parsed.get("stress_conditions", [])) or "stress tolerance"
    location = parsed.get("location", "")
    # Include location in query so we get India-specific research where possible
    query    = f"{crop} {stress} {location}".strip()

    try:
        research = fetch_research(query)
        logger.info("[research_node] Fetched %d insights.", len(research))
        return {**state, "research": research}
    except Exception as exc:
        logger.warning("[research_node] %s — using empty research.", exc)
        return {**state, "research": []}


def build_node(state: PipelineState) -> PipelineState:
    spec = build_spec(
        state.get("parsed") or {},
        state.get("traits") or [],
        state.get("research") or [],
    )
    return {**state, "spec": spec}


def validate_node(state: PipelineState) -> PipelineState:
    validated = validate_spec(state.get("spec") or {})
    if validated:
        # Export to disk / send to simulation team
        pipeline_id = state.get("pipeline_id", "unknown")
        export_spec(validated, pipeline_id)
        return {**state, "validated": validated, "error": None}
    return {
        **state,
        "error": "Spec validation failed.",
        "retries": state["retries"] + 1,
    }


# ── Routing ───────────────────────────────────────────────────────────────────

def _route(state: PipelineState) -> str:
    if state.get("validated"):
        return "done"
    if state["retries"] < settings.max_retries:
        logger.info("[workflow] Retry %d/%d", state["retries"], settings.max_retries)
        return "retry"
    logger.error("[workflow] Max retries reached.")
    return "done"


# ── Graph ─────────────────────────────────────────────────────────────────────

def _build_graph():
    g = StateGraph(PipelineState)
    g.add_node("parse",    parse_node)
    g.add_node("search",   search_node)
    g.add_node("research", research_node)
    g.add_node("build",    build_node)
    g.add_node("validate", validate_node)

    g.set_entry_point("parse")
    g.add_edge("parse",    "search")
    g.add_edge("search",   "research")
    g.add_edge("research", "build")
    g.add_edge("build",    "validate")
    g.add_conditional_edges("validate", _route, {"retry": "parse", "done": END})
    return g.compile()


_graph = None


def run_pipeline(prompt: str) -> dict:
    global _graph
    if _graph is None:
        _graph = _build_graph()

    pipeline_id = str(uuid.uuid4())

    result = _graph.invoke({
        "prompt":    prompt,
        "pipeline_id": pipeline_id,      
        "parsed":    None,
        "traits":    [],
        "research":  [],
        "spec":      None,
        "validated": None,
        "error":     None,
        "retries":   0,
    })

    return {
        "pipeline_id": pipeline_id,
        "spec":        result.get("validated"),
        "error":       result.get("error") if not result.get("validated") else None,
    }
