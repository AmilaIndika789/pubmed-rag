"""
Retrieve relevant chunks and generate grounded answers with Gemini.
"""

from __future__ import annotations

import os
from typing import Any
import inspect

from google import genai
from google.genai import errors

from utils import load_env
from rag.prompts import format_context, get_system_prompt
from vectordb.store import search

"""
    Available models models in free tier:
        * gemini-3.1-flash-lite-preview (500 Requests/Day)
        * gemini-3-flash-preview        (20 Requests/Day)
        * gemini-2.5-flash              (20 Requests/Day)
        * gemini-2.5-flash-lite         (20 Requests/Day)
"""
GEMINI_MODEL_NAME = "gemini-3.1-flash-lite-preview"

# Distance threshold to measure closeness of ChromaDB vectors
MAX_DISTANCE_THRESHOLD = 1.0
MIN_CLOSE_CHUNKS = 4


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

    try:
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

    except errors.ClientError as e:
        print(f"Client Error: {e}")
        return "There was an error processing the request."

    except errors.APIError as e:
        print(f"API Error: {e}")
        return "Service temporarily unavailable. Please try again later."

    except Exception as e:
        print(f"Unexpected Error: {e}")
        return "An unexpected error occurred."

    return response.text.strip() if response.text else ""


def has_sufficient_context(
    results: list[dict[str, Any]],
    max_distance_threshold: float = 1.3,
    min_good_chunks: int = 2,
) -> bool:
    """
    A simple sufficiency heuristic.

    Rules:
    - no results -> insufficient
    - at least 2 decent chunks is stronger evidence that retrieval is on-topic
    - L2/Euclidean distance indicate closeness
        * 0.0 - Same embedding
        * around 0.3 - 0.7 : Strong semantic match
        * around 0.7 - 1.1 : Maybe relevant/weak match
        * around 1.1 - 1.5 : Often weak/noisy match
        * up toward 2.0    : Mismatch
    """
    if not results:
        return False

    print(
        f"Results: {[{'pmid': result['pmid'], 'score': result['score']} for result in results]}"
    )

    good_chunks = [
        result
        for result in results
        if result.get("score") is not None and result["score"] <= max_distance_threshold
    ]

    return len(good_chunks) >= min_good_chunks


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

    if not has_sufficient_context(
        retrieved_chunks,
        max_distance_threshold=MAX_DISTANCE_THRESHOLD,
        min_good_chunks=MIN_CLOSE_CHUNKS,
    ):
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
