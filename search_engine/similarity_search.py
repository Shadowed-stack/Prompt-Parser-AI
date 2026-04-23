from search_engine.vector_store import collection
from search_engine.embeddings import embed
from shared.config import settings

def search_traits(query: str):
    q = embed(query).tolist()

    res = collection.query(
        query_embeddings=[q],
        n_results=settings.search_top_k
    )

    return res["documents"][0]
