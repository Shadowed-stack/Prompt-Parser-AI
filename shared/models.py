"""
Pydantic models for PRANAG-AI pipeline.

Changes over previous version:
  • Spec.crop and Spec.location are now Optional (default None)
    — agriculture queries still populate them via domain_parameters
    — non-agriculture queries (physics, chemistry etc.) don't have crop/location
    — Pydantic no longer rejects multi-domain specs that lack these fields
  • Spec.domain_weights and Spec.domain_parameters added
    — carries the full multi-domain output from domain_router + prompt_parser
  • ParsedPrompt kept for backward compatibility but no longer used by pipeline
    (pipeline now uses domain_parameters dict instead)
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


# ── Intermediate pipeline models ──────────────────────────────────────────────

class ParsedPrompt(BaseModel):
    """Kept for backward compatibility. Pipeline now uses domain_parameters."""
    crop:              Optional[str]   = None
    location:          Optional[str]   = None
    temperature:       Optional[float] = None
    humidity:          Optional[float] = None
    rainfall:          Optional[float] = None
    soil_type:         Optional[str]   = None
    stress_conditions: List[str]       = Field(default_factory=list)
    target_traits:     List[str]       = Field(default_factory=list)
    constraints:       Dict[str, Any]  = Field(default_factory=dict)


class Trait(BaseModel):
    trait_name:       str
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    domain:           str   = "biology"


class ResearchInsight(BaseModel):
    title:       str
    key_finding: str
    relevance:   float          = Field(..., ge=0.0, le=1.0)
    source:      str            = "semantic_scholar"
    url:         Optional[str]  = None


# ── Final validated output ────────────────────────────────────────────────────

class Spec(BaseModel):
    """
    The canonical spec.json handed off to the simulation team.

    Multi-domain fields (always present):
      domain_weights      — {domain: weight} from domain_router
      domain_parameters   — {domain: {param: value}} from prompt_parser

    Agriculture fields (present for crop queries, None otherwise):
      crop, location, temperature, humidity, rainfall, soil_type,
      stress_conditions, target_traits, constraints
    """

    # ── Multi-domain fields (always present) ──────────────────────────────────
    domain_weights:    Dict[str, float]        = Field(default_factory=dict)
    domain_parameters: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # ── Agriculture fields (optional — None for non-crop queries) ─────────────
    crop:              Optional[str]   = None
    location:          Optional[str]   = None
    temperature:       Optional[float] = None
    humidity:          Optional[float] = None
    rainfall:          Optional[float] = None
    soil_type:         Optional[str]   = None
    stress_conditions: List[str]       = Field(default_factory=list)
    target_traits:     List[str]       = Field(default_factory=list)
    constraints:       Dict[str, Any]  = Field(default_factory=dict)

    # ── Search + research results (always present) ────────────────────────────
    retrieved_traits:  List[str]           = Field(default_factory=list)
    scientific_basis:  List[str]           = Field(default_factory=list)
    research_titles:   List[str]           = Field(default_factory=list)
    research_sources:  List[str]           = Field(default_factory=list)
    research_years:    List[Optional[int]] = Field(default_factory=list)

    # ── Metadata ──────────────────────────────────────────────────────────────
    confidence:        float = Field(..., ge=0.0, le=1.0)
    pipeline_version:  str   = "1.0.0"
    generated_at:      str   = ""