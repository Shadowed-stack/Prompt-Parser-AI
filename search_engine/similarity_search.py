"""
Similarity search over ChromaDB.

BUG FIXED: original code passed  n_results=[settings.search_top_k]  (a list),
           ChromaDB requires an int.  Fixed to  n_results=settings.search_top_k.
"""
from search_engine.vector_store import get_collection
from search_engine.embeddings import embed
from shared.config import settings


def search_traits(query: str, top_k: int | None = None) -> list[str]:
    """
    Returns the top-k most similar trait descriptions for *query*.

    Args:
        query:  Natural language query, e.g. "wheat drought resistance Jodhpur".
        top_k:  Override the global setting.search_top_k if provided.

    Returns:
        List of trait description strings, best match first.
    """
    k = top_k if top_k is not None else settings.search_top_k
    col = get_collection()

    # Guard: if collection is empty, return empty list (avoids ChromaDB error)
    if col.count() == 0:
        return []

    vector = embed(query).tolist()
    results = col.query(
        query_embeddings=[vector],
        n_results=min(k, col.count()),   # ← FIX: int, not list
        include=["documents", "distances"],
    )
    return results["documents"][0]       # list of matched trait strings


def search_traits_with_scores(query: str, top_k: int | None = None) -> list[dict]:
    """
    Same as search_traits but also returns similarity scores (1 − cosine distance).
    """
    k = top_k if top_k is not None else settings.search_top_k
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
            "trait": doc,
            "score": round(1.0 - dist, 4),   # cosine distance → similarity
            "domain": meta.get("domain", "unknown"),
        }
        for doc, dist, meta in zip(docs, distances, metas)
    ]
