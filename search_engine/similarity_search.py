"""
Similarity search over ChromaDB — Jay's Task 3 + Task 7

Changes over GitHub version:
  • embed_cached() used instead of embed() — repeated queries return
    from memory without re-running the model (Task 7)
  • Timing measurement added — logs warning if query exceeds 50ms target
    specified in WorkflowB.pdf (Task 3)
  • Empty store warning instead of silent empty return
  • search_traits_with_scores() also timed and returns domain + tags metadata
  • Tags-based search added for Parquet data: when crop is known,
    the search filters by tags containing the crop name since the
    universal_index_final.parquet uses tags (not a crop column) for
    crop-specific filtering
"""
from __future__ import annotations

import logging
import time

from search_engine.vector_store import get_collection
from search_engine.embeddings import embed_cached
from shared.config import settings

logger = logging.getLogger(__name__)

# Target from WorkflowB.pdf spec: <50ms per search query
_SEARCH_TARGET_MS = 50.0


def search_traits(
    query: str,
    top_k: int | None = None,
    crop: str | None = None,
) -> list[str]:
    """
    Returns top-k most similar trait/entity descriptions for *query*.

    Jay Task 3 — Similarity Search API:
      Target: <50ms per query. Logs a warning when exceeded.

    Strategy:
      1. If crop is known, try tag-filtered search first to get
         crop-specific results (the Parquet uses tags like
         'agriculture,crop,wheat' for crop identification).
      2. Fill remaining slots from global search (deduplicated).
      3. Crop-specific results always appear first in the returned list.

    Args:
        query:  Natural language query string.
        top_k:  Number of results. Defaults to settings.search_top_k.
        crop:   Crop name for tag-boosted filtering (e.g. "wheat", "rice").
                MUST be passed from workflow.py as parsed.get("crop").

    Returns:
        List of description strings, crop-specific first then global.
    """
    start = time.perf_counter()

    k   = top_k if top_k is not None else settings.search_top_k
    col = get_collection()

    if col.count() == 0:
        logger.warning(
            "[similarity_search] Vector store is empty. "
            "Run populate() or load_from_parquet() first."
        )
        return []

    # Use cached embedding (Task 7 — skip model on repeated queries)
    vector = list(embed_cached(query))

    crop_docs: list[str] = []

    # ── Step 1: crop-specific search using tag filter ─────────────────────────
    # The Parquet file uses the 'tags' metadata field for crop identification.
    # The seed traits use the 'crop' metadata field.
    # We try both approaches so the search works correctly in both modes.
    if crop and crop.lower() not in ("unknown", "", "general"):
        crop_lower = crop.lower()

        # Try tag-based filtering (works for Parquet data)
        try:
            crop_res = col.query(
                query_embeddings=[vector],
                n_results=min(k, col.count()),
                where={"$or": [
                    {"tags":  {"$contains": crop_lower}},
                    {"crop":  {"$in": [crop_lower, "general"]}},
                ]},
                include=["documents"],
            )
            crop_docs = crop_res["documents"][0]
        except Exception:
            # ChromaDB $contains may not be supported in all versions
            # Fall back to crop= metadata filter only
            try:
                crop_res = col.query(
                    query_embeddings=[vector],
                    n_results=min(k, col.count()),
                    where={"crop": {"$in": [crop_lower, "general"]}},
                    include=["documents"],
                )
                crop_docs = crop_res["documents"][0]
            except Exception:
                crop_docs = []

    # ── Step 2: global search ─────────────────────────────────────────────────
    global_res = col.query(
        query_embeddings=[vector],
        n_results=min(k, col.count()),
        include=["documents"],
    )
    global_docs = global_res["documents"][0]

    # ── Step 3: merge — crop-specific first, then global (deduplicated) ───────
    seen: set[str] = set()
    merged: list[str] = []
    for doc in crop_docs + global_docs:
        if doc not in seen:
            seen.add(doc)
            merged.append(doc)
        if len(merged) >= k:
            break

    # ── Step 4: timing check ──────────────────────────────────────────────────
    elapsed_ms = (time.perf_counter() - start) * 1000
    if elapsed_ms > _SEARCH_TARGET_MS:
        logger.warning(
            "[similarity_search] Slow query: %.1fms (target <%dms) for: '%s'",
            elapsed_ms, _SEARCH_TARGET_MS, query[:60],
        )
    else:
        logger.debug(
            "[similarity_search] %.1fms for: '%s'",
            elapsed_ms, query[:60],
        )

    return merged


def search_traits_with_scores(
    query: str,
    top_k: int | None = None,
    crop: str | None = None,
) -> list[dict]:
    """
    Same as search_traits but also returns similarity scores and metadata.

    Returns list of dicts:
        trait  — entity description string
        score  — cosine similarity (0.0 to 1.0, higher = more similar)
        domain — scientific domain (biology/chemistry/physics/materials/environment)
        tags   — comma-separated tags from the Parquet file
    """
    start = time.perf_counter()

    k   = top_k if top_k is not None else settings.search_top_k
    col = get_collection()

    if col.count() == 0:
        logger.warning("[similarity_search] Vector store is empty.")
        return []

    vector = list(embed_cached(query))

    results = col.query(
        query_embeddings=[vector],
        n_results=min(k, col.count()),
        include=["documents", "distances", "metadatas"],
    )

    docs      = results["documents"][0]
    distances = results["distances"][0]
    metas     = results.get("metadatas", [[{}] * len(docs)])[0]

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.debug("[similarity_search] with_scores: %.1fms for: '%s'", elapsed_ms, query[:60])

    return [
        {
            "trait":  doc,
            "score":  round(1.0 - dist, 4),
            "domain": meta.get("domain", "unknown"),
            "tags":   meta.get("tags",   ""),
        }
        for doc, dist, meta in zip(docs, distances, metas)
    ]
