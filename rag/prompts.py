"""
Prompt templates and context formatting for RAG.
"""

from __future__ import annotations

import inspect
from typing import Any

SYSTEM_PROMPT_V1 = """
You are a medical literature QA assistant.

Answer the user's question using only the provided context.
Cite PMID numbers when making important claims.
If the context is insufficient, say that you do not have enough information.
Be concise and evidence-based.
""".strip()


SYSTEM_PROMPT_V2 = """
You are a medical literature QA assistant.

Use only the retrieved article context provided to answer the user's question.
Do not use outside knowledge.
For every important medical claim, cite the supporting PMID(s).
If the context is insufficient, respond exactly:
"I don't have enough information from the retrieved articles."
If the retrieved articles disagree or show mixed findings, say so clearly.
Be concise, cautious, and evidence-based.
""".strip()


def get_system_prompt(version: str = "v2") -> str:
    """Return the selected system prompt."""
    if version == "v1":
        return SYSTEM_PROMPT_V1
    return SYSTEM_PROMPT_V2


def format_context(chunks: list[dict[str, Any]]) -> str:
    """
    Turn retrieved chunks into a formatted context block for the LLM.
    """
    parts: list[str] = []

    for idx, chunk in enumerate(chunks, start=1):
        content = inspect.cleandoc(
            f"""
            [Source {idx}]
            PMID: {chunk.get("pmid", "")}
            Title: {chunk.get("title", "")}
            Section: {chunk.get("section", "")}
            Content: {chunk.get("text", "")}
            """
        )
        parts.append(content)

    return "\n\n".join(parts)
