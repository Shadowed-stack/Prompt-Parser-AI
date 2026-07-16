"""
Research Fetcher — fetches real scientific papers from OpenAlex API.

Changes over previous version:
  • _fallback_insights() no longer uses agriculture-specific language
    ("Agronomic Resilience", "environmental stressors") — now uses
    neutral topic-based text that works for any domain.
"""
import requests
import logging
from typing import List, Dict, Any
from functools import lru_cache
from shared.config import settings

logger = logging.getLogger(__name__)


def reconstruct_abstract(inverted_index: Dict[str, List[int]]) -> str:
    """OpenAlex returns abstracts as an inverted index to save space.
    This helper stitches the words back into a readable paragraph."""
    if not inverted_index:
        return "Detailed findings available in the full text."
    try:
        max_idx = max(pos for positions in inverted_index.values() for pos in positions)
        words   = [""] * (max_idx + 1)
        for word, positions in inverted_index.items():
            for pos in positions:
                words[pos] = word
        return " ".join(words).strip()
    except Exception:
        return "Abstract processing error."


@lru_cache(maxsize=128)
def fetch_research(query: str) -> List[Dict[str, Any]]:
    """Fetches real scientific papers from the completely free OpenAlex API."""
    logger.info("[research_fetcher] Fetching OpenAlex research for: '%s'", query)

    url   = "https://api.openalex.org/works"
    limit = settings.research_results_limit

    params = {
        "search":   query,
        "per_page": limit,
        "filter":   "has_abstract:true",
        "sort":     "relevance_score:desc",
        "mailto":   "a@gmail.com",
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("results", []):
            full_abstract = reconstruct_abstract(item.get("abstract_inverted_index"))
            sentences     = full_abstract.split(". ")
            key_finding   = sentences[0] + "." if sentences else full_abstract

            insight = {
                "title":       item.get("display_name", "Unknown Title"),
                "key_finding": key_finding[:250] + "..." if len(key_finding) > 250 else key_finding,
                "relevance":   0.85,
                "source":      "OpenAlex",
                "url":         item.get("doi") or item.get("id") or "No URL available",
            }
            results.append(insight)

        return results

    except Exception as e:
        logger.error("[research_fetcher] OpenAlex API failed: %s", e)
        return _fallback_insights(query)


def _fallback_insights(query: str) -> List[Dict[str, Any]]:
    """Neutral offline fallback — works for any domain, not just agriculture."""
    words = query.split()
    topic = " ".join(words[:3]).title() if words else "This Topic"

    return [
        {
            "title":       f"Research Overview: {topic}",
            "key_finding": "Peer-reviewed studies report significant findings related to the specified parameters and conditions.",
            "relevance":   0.5,
            "source":      "offline_fallback",
            "url":         "offline_fallback",
        }
    ]