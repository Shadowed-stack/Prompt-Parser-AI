"""
Similarity search — parquet-native numpy vector store, all-domain version.

WHAT CHANGED vs previous version
──────────────────────────────────
  • _RELEVANT_DOMAINS now includes ALL 14 domains from domain.txt
    (was only {"biology", "environment"} — blocked chemistry/physics/etc.)
  • _build_masks() now uses active_domains parameter so search_traits()
    can optionally restrict to only the domains the LLM detected as relevant
  • search_traits_with_scores() boosting updated for all domains
  • crop/agriculture tag boosts preserved for backward compatibility

PUBLIC API (identical signatures — nothing else changes)
──────────
  search_traits(query, top_k, crop, active_domains)         → list[str]
  search_traits_with_scores(query, top_k, crop, active_domains) → list[dict]
"""

import logging
import time

import numpy as np

from search_engine.embeddings import embed_cached
from shared.config import settings

logger = logging.getLogger(__name__)

_SEARCH_TARGET_MS = 50.0

# ALL domains from domain.txt — no longer restricted to biology/environment
_ALL_DOMAINS = {
    "biology",
    "chemistry",
    "physics",
    "materials_science",
    "earth_environment",
    "medicine",
    "engineering",
    "computer_science_ai",
    "mathematics",
    "quant_finance",
    "astronomy_space",
    "scientific_programming",
    "human_social",
    "economics",
    # parquet-native domain names (may differ slightly from domain.txt keys)
    "materials",
    "environment",
    "general",
}

# Domain boost values — biology/environment kept for backward compat,
# all other domains get a neutral boost so they're not penalised
_DOMAIN_BOOSTS = {
    "biology":              0.20,
    "environment":          0.20,
    "earth_environment":    0.20,
    "chemistry":            0.20,
    "physics":              0.20,
    "materials_science":    0.20,
    "materials":            0.20,
    "medicine":             0.20,
    "engineering":          0.20,
    "computer_science_ai":  0.20,
    "mathematics":          0.20,
    "quant_finance":        0.20,
    "astronomy_space":      0.20,
    "scientific_programming": 0.20,
    "human_social":         0.20,
    "economics":            0.20,
    "general":              0.10,
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _is_store_ready() -> bool:
    from search_engine.vector_store import _df as df, _vectors as vecs
    return df is not None and vecs is not None and len(df) > 0


def _raw_search(
    query_vec: np.ndarray,
    top_k: int,
    domain_mask: np.ndarray | None = None,
    crop_mask: np.ndarray | None = None,
) -> list[dict]:
    """
    Core search — single numpy matmul, returns top_k dicts with scores.
    """
    from search_engine.vector_store import _df as df, _vectors as vecs

    # Combine masks
    if domain_mask is not None and crop_mask is not None:
        mask = domain_mask & crop_mask
    elif domain_mask is not None:
        mask = domain_mask
    elif crop_mask is not None:
        mask = crop_mask
    else:
        mask = None

    # Fall back progressively if mask is too restrictive
    if mask is not None and mask.sum() == 0:
        logger.debug("[similarity_search] Combined mask matched 0 rows — falling back to domain mask.")
        mask = domain_mask

    if mask is not None and mask.sum() == 0:
        logger.debug("[similarity_search] Domain mask matched 0 rows — running unfiltered.")
        mask = None

    actual_k = min(top_k, int(mask.sum()) if mask is not None else len(vecs))

    if mask is not None:
        scores_full = np.full(len(vecs), -2.0, dtype=np.float32)
        scores_full[mask] = vecs[mask] @ query_vec
    else:
        scores_full = vecs @ query_vec

    top_idx = np.argpartition(scores_full, -actual_k)[-actual_k:]
    top_idx = top_idx[np.argsort(scores_full[top_idx])[::-1]]

    results = []
    for idx in top_idx:
        row = df.iloc[int(idx)]
        results.append({
            "trait":     row.get("_text", ""),
            "score":     float(scores_full[idx]),
            "domain":    str(row.get("domain", "")).lower(),
            "tags":      str(row.get("tags", "")).lower(),
            "entity_id": str(row.get("entity_id", "")),
        })
    return results


def _build_masks(
    crop_lower: str | None,
    active_domains: set[str] | None = None,
):
    """
    Build pandas boolean masks for domain and crop filtering.

    Args:
        crop_lower     : lowercase crop name or None
        active_domains : set of domain names from domain_router (LLM-detected).
                         If None, all known domains are included.

    Returns:
        (domain_mask, crop_mask) — both numpy arrays (N,) or None
    """
    from search_engine.vector_store import _df as df

    if df is None:
        return None, None

    domain_col = df["domain"].str.lower()

    # If LLM gave us active domains, restrict to those.
    # Otherwise allow everything in _ALL_DOMAINS (no agriculture restriction).
    if active_domains:
        # Normalise domain.txt keys → parquet domain names where needed
        normalised = _normalise_domains(active_domains)
        domain_mask = domain_col.isin(normalised).to_numpy()
    else:
        domain_mask = domain_col.isin(_ALL_DOMAINS).to_numpy()

    # If domain mask is still empty (parquet uses different domain strings),
    # fall back to no domain filtering at all
    if domain_mask.sum() == 0:
        logger.debug("[similarity_search] Domain mask empty — running unfiltered.")
        domain_mask = None

    # Crop mask — only built when crop is provided (agriculture queries)
    crop_mask = None
    if crop_lower:
        tags_col     = df["tags"].fillna("").str.lower()
        crop_in_tags = tags_col.str.contains(crop_lower, regex=False)
        if "crop" in df.columns:
            crop_col   = df["crop"].fillna("").str.lower()
            crop_match = (crop_col == crop_lower) | (crop_col == "general")
        else:
            crop_match = crop_in_tags
        crop_mask = (crop_in_tags | crop_match).to_numpy()

    return domain_mask, crop_mask


def _normalise_domains(active_domains: set[str]) -> set[str]:
    """
    Map domain.txt section names → parquet domain column values.

    domain.txt uses:  earth_environment, materials_science, computer_science_ai
    parquet uses:     environment, materials, (no direct equivalent for CS)

    We include both so queries match regardless of which naming convention
    the parquet data uses.
    """
    mapping = {
        "earth_environment":    {"environment", "earth_environment"},
        "materials_science":    {"materials", "materials_science"},
        "computer_science_ai":  {"computer_science_ai", "general"},
        "biology":              {"biology"},
        "chemistry":            {"chemistry"},
        "physics":              {"physics"},
        "medicine":             {"medicine", "biology"},   # parquet may store as biology
        "engineering":          {"engineering", "materials"},
        "mathematics":          {"mathematics", "general"},
        "quant_finance":        {"quant_finance", "general"},
        "astronomy_space":      {"astronomy_space", "physics"},
        "scientific_programming": {"scientific_programming", "general"},
        "human_social":         {"human_social", "general"},
        "economics":            {"economics", "general"},
        "general":              {"general"},
    }
    result = set()
    for d in active_domains:
        result.update(mapping.get(d, {d}))
    return result


# ── Public API ────────────────────────────────────────────────────────────────

def search_traits(
    query: str,
    top_k: int | None = None,
    crop: str | None = None,
    active_domains: dict | None = None,   # domain_weights from domain_router
) -> list[str]:
    """
    Fast similarity search returning top-k trait description strings.

    Args:
        query          : Natural-language query string
        top_k          : Max results (defaults to settings.search_top_k)
        crop           : Optional crop name (agriculture queries only)
        active_domains : Dict of {domain: weight} from domain_router.
                         When provided, search is restricted to those domains.
                         When None, all domains are searched.

    Returns:
        List of trait/entity description strings, highest-scoring first
    """
    if not _is_store_ready():
        logger.warning(
            "[similarity_search] Vector store is empty. "
            "Run populate() or load_from_parquet() first."
        )
        return []

    k          = top_k if top_k is not None else settings.search_top_k
    crop_lower = crop.lower() if crop and crop.lower() not in ("unknown", "", "general") else None
    domain_set = set(active_domains.keys()) if active_domains else None

    query_vec = np.array(embed_cached(query), dtype=np.float32)
    start     = time.perf_counter()

    domain_mask, crop_mask = _build_masks(crop_lower, active_domains=domain_set)
    results = _raw_search(query_vec, top_k=k, domain_mask=domain_mask, crop_mask=crop_mask)

    # Top up from domain-filtered (no crop filter) if not enough results
    if crop_lower and len(results) < k:
        seen  = {r["trait"] for r in results}
        extra = _raw_search(query_vec, top_k=k, domain_mask=domain_mask, crop_mask=None)
        for r in extra:
            if r["trait"] not in seen:
                results.append(r)
                seen.add(r["trait"])
            if len(results) >= k:
                break

    elapsed_ms = (time.perf_counter() - start) * 1000
    if elapsed_ms > _SEARCH_TARGET_MS:
        logger.warning(
            "[similarity_search] Slow query: %.1fms (target <%.0fms) for: '%s'",
            elapsed_ms, _SEARCH_TARGET_MS, query[:60],
        )
    else:
        logger.debug("[similarity_search] %.1fms for '%s'", elapsed_ms, query[:60])

    return [r["trait"] for r in results[:k]]


def search_traits_with_scores(
    query: str,
    top_k: int | None = None,
    crop: str | None = None,
    active_domains: dict | None = None,
) -> list[dict]:
    """
    Similarity search with relevance scores and metadata.

    Scoring:
    ────────
    Base score = cosine similarity (0→1)
    Domain boost: all active domains get +0.20 (equal treatment)
    Tag boosts (agriculture): crop +0.20 | agriculture +0.20 | plant +0.15 | soil +0.10

    Returns:
        List of dicts sorted by boosted score descending:
        [{"trait": str, "score": float, "domain": str, "tags": str}, ...]
    """
    if not _is_store_ready():
        logger.warning("[similarity_search] Vector store is empty.")
        return []

    k          = top_k if top_k is not None else settings.search_top_k
    crop_lower = crop.lower() if crop and crop.lower() not in ("unknown", "", "general") else None
    domain_set = set(active_domains.keys()) if active_domains else None

    query_vec = np.array(embed_cached(query), dtype=np.float32)
    start     = time.perf_counter()

    domain_mask, crop_mask = _build_masks(crop_lower, active_domains=domain_set)

    fetch_k = min(k * 3, 500)
    raw     = _raw_search(query_vec, top_k=fetch_k, domain_mask=domain_mask, crop_mask=crop_mask)

    if crop_lower and len(raw) < fetch_k:
        seen  = {r["trait"] for r in raw}
        extra = _raw_search(query_vec, top_k=fetch_k, domain_mask=domain_mask, crop_mask=None)
        for r in extra:
            if r["trait"] not in seen:
                raw.append(r)
                seen.add(r["trait"])

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.debug("[similarity_search] with_scores: %.1fms for '%s'", elapsed_ms, query[:60])

    ranked: list[dict] = []
    seen:   set[str]   = set()

    for item in raw:
        trait = item["trait"]
        if trait in seen:
            continue
        seen.add(trait)

        score  = item["score"]
        domain = item["domain"]
        tags   = item["tags"]

        # Domain boost — equal for all domains
        score += _DOMAIN_BOOSTS.get(domain, 0.10)

        # Tag boosts — agriculture specific, preserved for backward compat
        if "crop"        in tags: score += 0.20
        if "agriculture" in tags: score += 0.20
        if "plant"       in tags: score += 0.15
        if "soil"        in tags: score += 0.10

        ranked.append({
            "trait":  trait,
            "score":  round(score, 4),
            "domain": domain,
            "tags":   tags,
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:k]