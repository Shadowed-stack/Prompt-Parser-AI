"""
PRANAG-AI Prompt-Parser Workflow  (LangGraph StateGraph)

Changes over previous version:
  • search_node — fixed NameError: domain_weights was referenced before
    assignment. Now correctly reads from state:
      domain_weights = state.get("domain_weights") or {}
  • Everything else unchanged.
"""
import uuid
import logging
from orchestrator.domain_router import score_domains
from typing import TypedDict, Optional
from shared.profiler import time_block
from langgraph.graph import StateGraph, END

from shared.config import settings
from orchestrator.prompt_parser import parse_prompt
from orchestrator.research_fetcher import fetch_research
from orchestrator.spec_builder import build_spec
from orchestrator.output_validator import validate_spec
from orchestrator.output_exporter import export_spec
from search_engine.similarity_search import search_traits

logger = logging.getLogger(__name__)


# ── State ─────────────────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    prompt:         str
    pipeline_id:    str
    domain_weights: Optional[dict]
    parsed:         Optional[dict]
    traits:         list
    research:       list
    spec:           Optional[dict]
    validated:      Optional[dict]
    error:          Optional[str]
    retries:        int


# ── Nodes ─────────────────────────────────────────────────────────────────────

def domain_node(state: PipelineState) -> PipelineState:
    """Layer 0: score all domains, keep active ones."""
    with time_block("Pipeline-DomainNode"):
        weights = score_domains(state["prompt"])
        return {"domain_weights": weights}


def parse_node(state: PipelineState) -> PipelineState:
    with time_block("Pipeline-ParseNode"):
        try:
            parsed = parse_prompt(state["prompt"], state.get("domain_weights") or {})
            return {"parsed": parsed, "error": None}
        except Exception as exc:
            logger.error("[parse_node] Unexpected error: %s", exc)
            return {"parsed": None, "error": str(exc), "retries": state["retries"] + 1}


def search_node(state: PipelineState) -> PipelineState:
    with time_block("Pipeline-SearchNode"):
        parsed = state.get("parsed") or {}

        # FIX: read domain_weights from state (was undefined local variable before)
        domain_weights = state.get("domain_weights") or {}

        # Build query from domain_parameters
        domain_params = parsed.get("domain_parameters", {})
        query_parts   = []
        for domain, params in domain_params.items():
            if isinstance(params, dict):
                query_parts.extend(str(v) for v in params.values() if v)
        query = " ".join(query_parts).strip() or state["prompt"]

        try:
            traits = search_traits(
                query,
                crop=None,
                active_domains=domain_weights,
            )
            return {"traits": traits}
        except Exception as exc:
            logger.warning("[search_node] Search failed: %s", exc)
            return {"traits": []}


def research_node(state: PipelineState) -> PipelineState:
    with time_block("Pipeline-ResearchNode"):
        parsed         = state.get("parsed") or {}
        domain_weights = state.get("domain_weights") or {}
        domain_params  = parsed.get("domain_parameters", {})

        if domain_weights:
            top_domain = max(domain_weights, key=domain_weights.get)
            top_params = domain_params.get(top_domain, {})
            if isinstance(top_params, dict):
                query = " ".join(str(v) for v in top_params.values() if v).strip()
            else:
                query = ""
        else:
            query = state["prompt"]

        query = query or state["prompt"]

        try:
            research = fetch_research(query)
            return {"research": research}
        except Exception as exc:
            logger.warning("[research_node] %s", exc)
            return {"research": []}


def build_node(state: PipelineState) -> PipelineState:
    with time_block("Pipeline-BuildNode"):
        spec = build_spec(
            state.get("parsed") or {},
            state.get("traits") or [],
            state.get("research") or [],
        )
    return {"spec": spec}


def validate_node(state: PipelineState) -> PipelineState:
    with time_block("Pipeline-ValidateNode"):
        validated = validate_spec(state.get("spec") or {})
        if validated:
            pipeline_id = state.get("pipeline_id", "unknown")
            export_spec(validated, pipeline_id)
            return {"validated": validated, "error": None}
        return {
            "error":   "Spec validation failed.",
            "retries": state["retries"] + 1,
        }


# ── Routing ───────────────────────────────────────────────────────────────────

def _route(state: PipelineState) -> str:
    with time_block("Pipeline-_route"):
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
    g.add_node("domain",   domain_node)
    g.add_node("parse",    parse_node)
    g.add_node("search",   search_node)
    g.add_node("research", research_node)
    g.add_node("build",    build_node)
    g.add_node("validate", validate_node)

    g.set_entry_point("domain")
    g.add_edge("domain",  "parse")
    g.add_edge("parse",   "search")
    g.add_edge("parse",   "research")
    g.add_edge(["search", "research"], "build")
    g.add_edge("build",   "validate")
    g.add_conditional_edges("validate", _route, {"retry": "parse", "done": END})
    return g.compile()


_graph = None


def run_pipeline(prompt: str) -> dict:
    global _graph
    if _graph is None:
        _graph = _build_graph()

    pipeline_id = str(uuid.uuid4())

    result = _graph.invoke({
        "prompt":         prompt,
        "pipeline_id":    pipeline_id,
        "parsed":         None,
        "traits":         [],
        "research":       [],
        "domain_weights": None,
        "spec":           None,
        "validated":      None,
        "error":          None,
        "retries":        0,
    })

    return {
        "pipeline_id": pipeline_id,
        "spec":        result.get("validated"),
        "error":       result.get("error") if not result.get("validated") else None,
    }