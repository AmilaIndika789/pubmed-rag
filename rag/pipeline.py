"""
Retrieve relevant chunks and generate grounded answers with Gemini.
"""

from __future__ import annotations

import os
from typing import Any
import inspect

from google import genai

from utils import load_env
from rag.prompts import get_system_prompt
from vectordb.store import search

GEMINI_MODEL_NAME = "gemini-2.0-flash"


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


def build_user_prompt(question: str, context: str) -> str:
    """Build the user-facing prompt payload."""
    return inspect.cleandoc(
        f"""
        Context:
        {context}

        Question:
        {question}
        """
    ).strip()


def generate_with_gemini(
    question: str, context: str, prompt_version: str = "v2"
) -> str:
    """
    Generate an answer with Gemini.
    """
    client = get_gemini_client()
    system_prompt = get_system_prompt(prompt_version)
    user_prompt = build_user_prompt(question=question, context=context)

    response = client.models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=user_prompt,
        config={
            "system_instruction": system_prompt,
            "temperature": 0.2,
        },
    )

    return response.text.strip() if response.text else ""
