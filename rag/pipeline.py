"""
Retrieve relevant chunks and generate grounded answers with Gemini.
"""

from __future__ import annotations

import os
from typing import Any
import inspect

from google import genai

from utils import load_env
from rag.prompts import format_context, get_system_prompt
from vectordb.store import search


GEMINI_MODEL_NAME = "gemini-2.5-flash"


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


def has_sufficient_context(
    results: list[dict[str, Any]], max_distance_threshold: float = 1.3
) -> bool:
    """
    Very simple sufficiency heuristic.

    Rules:
    - no results -> insufficient
    - best result too far away -> insufficient
    """
    if not results:
        return False

    best_score = results[0]["score"]
    if best_score is None:
        return False

    return best_score <= max_distance_threshold


def remove_duplicate_sources(chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
    """
    Deduplicate source list by PMID.
    """
    unique: dict[str, dict[str, str]] = {}
    for chunk in chunks:
        pmid = chunk.get("pmid", "")
        if pmid not in unique:
            unique[pmid] = {
                "pmid": pmid,
                "title": chunk.get("title", ""),
            }
    return list(unique.values())


def answer_question(
    question: str,
    top_k: int = 5,
    strategy: str = "section",
    prompt_version: str = "v2",
) -> dict[str, Any]:
    """
    End-to-end RAG pipeline.

    Returns:
        Structured response payload for CLI or UI use.
    """
    retrieved_chunks = retrieve_chunks(query=question, top_k=top_k, strategy=strategy)

    if not has_sufficient_context(retrieved_chunks):
        return {
            "question": question,
            "answer": "I don't have enough information from the retrieved articles.",
            "sources": [],
            "used_context": retrieved_chunks,
            "prompt_version": prompt_version,
            "insufficient": True,
        }

    context = format_context(retrieved_chunks)
    answer = generate_with_gemini(
        question=question, context=context, prompt_version=prompt_version
    )

    return {
        "question": question,
        "answer": answer,
        "sources": remove_duplicate_sources(retrieved_chunks),
        "used_context": retrieved_chunks,
        "prompt_version": prompt_version,
        "insufficient": False,
    }
