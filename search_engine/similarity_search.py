"""
Similarity search over ChromaDB.

Improvements over v1:
  • Crop-boosted search: tries crop-specific filter first, falls back to global
  • Returns crop-relevant results at the top, not just cosine-closest
  • n_results bug fixed (int not list)
"""
from search_engine.vector_store import get_collection
from search_engine.embeddings import embed
from shared.config import settings


def search_traits(query: str, top_k: int | None = None, crop: str | None = None) -> list[str]:
    """
    Returns top-k most similar trait descriptions for *query*.

    Strategy:
      1. If *crop* is known and not 'unknown', fetch crop-specific results first.
      2. Fill remaining slots from global search (deduplicated).
      3. This ensures sugarcane queries return sugarcane traits, not wheat traits.

    Args:
        query:  Natural language query.
        top_k:  Override settings.search_top_k.
        crop:   Crop name for boosted filtering (e.g. "sugarcane").
    """
    k   = top_k if top_k is not None else settings.search_top_k
    col = get_collection()

    if col.count() == 0:
        return []

    vector = embed(query).tolist()

    crop_docs: list[str] = []

    # ── Step 1: crop-specific results ────────────────────────────────────────
    if crop and crop.lower() not in ("unknown", "", "general"):
        try:
            crop_res = col.query(
                query_embeddings=[vector],
                n_results=min(k, col.count()),
                where={"crop": {"$in": [crop.lower(), "general"]}},
                include=["documents"],
            )
            crop_docs = crop_res["documents"][0]
        except Exception:
            # ChromaDB where-filter may fail if metadata not set; fall through
            crop_docs = []

    # ── Step 2: global search to fill remaining slots ─────────────────────────
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

    return merged


def search_traits_with_scores(query: str, top_k: int | None = None, crop: str | None = None) -> list[dict]:
    """Same as search_traits but also returns similarity scores."""
    k   = top_k if top_k is not None else settings.search_top_k
    col = get_collection()

    if col.count() == 0:
        return []

    vector = embed(query).tolist()
    results = col.query(
        query_embeddings=[vector],
        n_results=min(k, col.count()),
        include=["documents", "distances", "metadatas"],
    )
    docs      = results["documents"][0]
    distances = results["distances"][0]
    metas     = results.get("metadatas", [[{}] * len(docs)])[0]

    return [
        {
            "trait":  doc,
            "score":  round(1.0 - dist, 4),
            "domain": meta.get("domain", "unknown"),
            "crop":   meta.get("crop", "general"),
        }
        for doc, dist, meta in zip(docs, distances, metas)
    ]
