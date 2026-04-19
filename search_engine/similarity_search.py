from .vector_store import collection
from .embeddings import embed

def search_traits(query, top_k=3):
    q = embed(query)
    res = collection.query(query_embeddings=[q], n_results=top_k)
    return res["documents"][0]
