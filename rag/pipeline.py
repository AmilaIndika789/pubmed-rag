"""
Retrieve relevant chunks and generate grounded answers with Gemini.
"""

from __future__ import annotations

import os
from typing import Any

from google import genai

from utils import load_env
from vectordb.store import search


def get_gemini_client() -> genai.Client:
    """Create Gemini client using environment variable."""
    load_env()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")
    return genai.Client(api_key=api_key)


def retrieve_chunks(
    query: str, top_k: int = 5, strategy: str = "section"
) -> list[dict[str, Any]]:
    """Retrieve top-k chunks from Chroma."""
    return search(query=query, top_k=top_k, strategy=strategy)
