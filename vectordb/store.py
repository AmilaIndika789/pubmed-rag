"""
Index chunks into ChromaDB and provide retrieval.

Run:
    python -m vectordb.store
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models import Collection

from vectordb.embeddings import embed_texts, embed_query


DB_DIR = "vectordb/db"

CHUNK_FILES = {
    "fixed": Path("chunking/data/chunks_fixed.jsonl"),
    "section": Path("chunking/data/chunks_section.jsonl"),
}

COLLECTION_NAMES = {
    "fixed": "pubmed_chunks_fixed",
    "section": "pubmed_chunks_section",
}


def get_chroma_client(persist_dir: str = DB_DIR) -> ClientAPI:
    """Return a persistent Chroma client."""
    return chromadb.PersistentClient(path=persist_dir)


def get_or_create_collection(collection_name: str) -> Collection.Collection:
    """Get or create a collection."""
    client = get_chroma_client()
    return client.get_or_create_collection(name=collection_name)


def load_chunks(path: Path) -> list[dict[str, Any]]:
    """Load JSONL chunks from disk."""
    chunks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def extract_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    """Extract metadata fields for Chroma storage."""
    return {
        "pmid": chunk["pmid"],
        "title": chunk["title"],
        "section": chunk["section"],
        "chunk_index": chunk["chunk_index"],
        "strategy": chunk["strategy"],
        "topic": chunk["topic"],
    }


def index_chunks(chunks: list[dict[str, Any]], collection_name: str) -> None:
    """
    Index chunks into Chroma.

    Stores:
    - ids
    - documents
    - embeddings
    - metadata
    """
    collection = get_or_create_collection(collection_name)

    ids = [chunk["chunk_id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [extract_metadata(chunk) for chunk in chunks]
    embeddings = embed_texts(documents)

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )


def main() -> None:
    strategy = "section"
    chunk_file = CHUNK_FILES[strategy]
    collection_name = COLLECTION_NAMES[strategy]

    chunks = load_chunks(chunk_file)
    print(f"Loaded {len(chunks)} chunks from {chunk_file}")

    index_chunks(chunks, collection_name=collection_name)
    print(f"Indexed into Chroma collection: {collection_name}")


if __name__ == "__main__":
    main()
