"""
Singleton embedding model.
All other modules import `embed()` from here.
"""
from sentence_transformers import SentenceTransformer
import numpy as np

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed(text: str) -> np.ndarray:
    """Return a normalised L2 embedding vector for *text*."""
    return _get_model().encode(text, normalize_embeddings=True)


def embed_batch(texts: list[str]) -> list[np.ndarray]:
    """Batch-encode for faster vector-store population."""
    return _get_model().encode(texts, normalize_embeddings=True, batch_size=64)
