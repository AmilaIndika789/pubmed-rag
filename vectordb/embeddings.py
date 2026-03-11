"""
Embedding helpers for chunks and queries.
"""

from __future__ import annotations

import os
from functools import lru_cache
from sentence_transformers import SentenceTransformer
import numpy as np

from utils import load_env


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load and cache the embedding model."""
    load_env()
    HF_TOKEN = os.getenv("HF_TOKEN")
    MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    return SentenceTransformer(MODEL_NAME, token=HF_TOKEN)


def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """
    Embed a batch of texts.

    Returns:
        List of embedding vectors as Python lists.
    """
    model = get_embedding_model()
    embeddings: np.ndarray = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([query])[0]
