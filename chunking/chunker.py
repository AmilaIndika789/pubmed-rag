"""
Create fixed-size and section-based chunks from PubMed articles.

Run:
    python -m chunking.chunker
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from utils import ensure_output_dir

INPUT_PATH = Path("ingest/data/JSON/pubmed_articles.json")
FIXED_OUTPUT_PATH = Path("chunking/data/chunks_fixed.jsonl")
SECTION_OUTPUT_PATH = Path("chunking/data/chunks_section.jsonl")

CHUNK_SIZE = 100  # In words
CHUNK_OVERLAP = 20  # In words


def load_articles(path: Path = INPUT_PATH) -> list[dict[str, Any]]:
    """Load raw article JSON."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text: str) -> str:
    """Normalize whitespace and trim text."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_words(text: str) -> list[str]:
    """Split text into words for simple word-based chunking."""
    return text.split()


def fixed_size_chunk(
    text: str, chunk_words: int = 150, overlap_words: int = 50
) -> list[str]:
    """
    Chunk text into overlapping windows by word count.

    This is a simple approximation of token-based chunking.
    """
    words = split_words(normalize_text(text))
    if not words:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = start + chunk_words
        chunk_words_list = words[start:end]
        chunk_text = " ".join(chunk_words_list).strip()
        if chunk_text:
            chunks.append(chunk_text)

        if end >= len(words):
            break

        start = end - overlap_words

    return chunks


def detect_labeled_sections(abstract: str) -> list[tuple[str, str]]:
    """
    Try to detect labeled abstract sections such as:
    Background, Methods, Results, Conclusion.

    Returns:
        List of tuples: [(section_name, section_text), ...]
    """
    section_headers: list[str] = [
        "Aim",
        "Purpose",
        "Background",
        "Introduction",
        "Methods",
        "Method",
        "Results",
        "Outcomes",
        "Findings",
        "Conclusion",
        "Conclusions",
        "Objective",
        "Objectives",
        "Benefits",
        "Recommendations",
        "Validity",
        "Summary",
    ]

    pattern = re.compile(
        rf"(?P<label>{'|'.join(section_headers)})\s*:\s*",
        flags=re.IGNORECASE,
    )

    matches = list(pattern.finditer(abstract))
    if not matches:
        return []

    sections: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        label = match.group("label").strip().lower()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(abstract)
        content = abstract[start:end].strip()
        if content:
            sections.append((label, content))

    return sections


def fallback_paragraph_sections(abstract: str) -> list[tuple[str, str]]:
    """
    Fallback splitter when no labeled sections are found.

    If there are paragraphs, use them. Otherwise return the whole abstract.
    """
    paragraphs = [p.strip() for p in abstract.split("\n\n") if p.strip()]
    if len(paragraphs) > 1:
        return [(f"paragraph_{i}", p) for i, p in enumerate(paragraphs)]

    clean_abstract = abstract.strip()
    return [("abstract", clean_abstract)] if clean_abstract else []


def section_based_chunk(article: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Create section-based chunks for one article.

    Each output chunk is already a structured record.
    """
    abstract = article.get("abstract", "")
    title = article.get("title", "")
    pmid = article.get("pmid", "")
    topic = article.get("topic", "")

    sections = detect_labeled_sections(abstract)
    if not sections:
        sections = fallback_paragraph_sections(abstract)

    chunks: list[dict[str, Any]] = []
    for idx, (section_name, section_text) in enumerate(sections):
        chunk_text = normalize_text(section_text)
        if not chunk_text:
            continue

        chunks.append(
            {
                "chunk_id": f"{pmid}_section_{idx}",
                "pmid": pmid,
                "title": title,
                "section": section_name,
                "chunk_index": idx,
                "strategy": "section",
                "topic": topic,
                "text": chunk_text,
            }
        )

    return chunks


def build_fixed_chunks(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Build fixed-size chunks for all articles.
    """
    all_chunks: list[dict[str, Any]] = []

    for article in articles:
        pmid = article.get("pmid", "")
        title = article.get("title", "")
        abstract = article.get("abstract", "")
        topic = article.get("topic", "")

        combined_text = normalize_text(f"{title}\n\n{abstract}")
        text_chunks = fixed_size_chunk(
            combined_text, chunk_words=CHUNK_SIZE, overlap_words=CHUNK_OVERLAP
        )

        for idx, chunk_text in enumerate(text_chunks):
            all_chunks.append(
                {
                    "chunk_id": f"{pmid}_fixed_{idx}",
                    "pmid": pmid,
                    "title": title,
                    "section": "abstract",
                    "chunk_index": idx,
                    "strategy": "fixed",
                    "topic": topic,
                    "text": chunk_text,
                }
            )

    return all_chunks


def build_section_chunks(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Build section-based chunks for all articles.
    """
    all_chunks: list[dict[str, Any]] = []
    for article in articles:
        all_chunks.extend(section_based_chunk(article))
    return all_chunks


def save_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Save records in JSONL format."""
    ensure_output_dir(path)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    """
    Main chunking pipeline.
    """
    articles = load_articles()

    fixed_chunks = build_fixed_chunks(articles)
    section_chunks = build_section_chunks(articles)

    save_jsonl(fixed_chunks, FIXED_OUTPUT_PATH)
    save_jsonl(section_chunks, SECTION_OUTPUT_PATH)

    print(f"Saved {len(fixed_chunks)} fixed chunks to {FIXED_OUTPUT_PATH}")
    print(f"Saved {len(section_chunks)} section chunks to {SECTION_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
