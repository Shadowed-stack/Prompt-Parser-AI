"""
Spec Builder

Combines:
  • parsed prompt  (from LLM)
  • retrieved traits  (from ChromaDB)
  • research insights  (from Semantic Scholar)

→ produces the raw dict that output_validator.py turns into a typed Spec.
"""
from shared.config import settings


def build_spec(
    parsed: dict,
    traits: list[str],
    research: list[dict],
) -> dict:
    """
    Merge all pipeline artefacts into a single spec dict.

    Args:
        parsed:    Output of prompt_parser.parse_prompt()
        traits:    List of trait description strings from similarity_search
        research:  List of ResearchInsight dicts from research_fetcher

    Returns:
        Raw dict matching the Spec schema in shared/models.py
    """
    # Confidence heuristic: penalise missing fields, reward more traits/research
    missing_penalty = sum([
        0.1 if not parsed.get("temperature") else 0,
        0.1 if not parsed.get("location") or parsed["location"] == "unknown" else 0,
        0.05 if not parsed.get("stress_conditions") else 0,
    ])
    trait_bonus    = min(len(traits) * 0.02, 0.1)
    research_bonus = min(len(research) * 0.02, 0.1)
    confidence     = round(
        max(0.0, min(1.0, 0.80 - missing_penalty + trait_bonus + research_bonus)),
        2,
    )

    return {
        "crop":              parsed.get("crop", "unknown"),
        "location":          parsed.get("location", "unknown"),
        "temperature":       parsed.get("temperature") or 25.0,
        "humidity":          parsed.get("humidity"),
        "rainfall":          parsed.get("rainfall"),
        "soil_type":         parsed.get("soil_type"),
        "stress_conditions": parsed.get("stress_conditions", []),
        "target_traits":     parsed.get("target_traits", []),
        "retrieved_traits":  traits,
        "scientific_basis":  [r["key_finding"] for r in research],
        "research_sources":  [r.get("url", "") for r in research],
        "constraints":       parsed.get("constraints", {}),
        "confidence":        confidence,
        "pipeline_version":  settings.pipeline_version,
    }
