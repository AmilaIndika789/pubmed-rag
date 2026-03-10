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
FIXED_OUTPUT_PATH = Path("chunking/data/JSON/chunks_fixed.jsonl")

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

    print(f"Fixed chunks: {json.dumps(fixed_chunks, indent=2)}")
    save_jsonl(fixed_chunks, FIXED_OUTPUT_PATH)

    print(f"Saved {len(fixed_chunks)} fixed chunks to {FIXED_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
