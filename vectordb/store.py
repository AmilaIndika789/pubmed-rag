"""
Index chunks into ChromaDB and provide retrieval.

Run:
    python -m vectordb.store --strategy section
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast, List

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models import Collection
from chromadb.api.types import Metadata, Embeddings

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
        metadatas=cast(List[Metadata], metadatas),
        embeddings=cast(Embeddings, embeddings),
    )


def ensure_collection_ready(strategy: str = "section") -> int:
    """
    Ensure a Chroma collection exists and is populated.

    Returns:
        Number of records in the ready collection.
    """
    collection_name = COLLECTION_NAMES[strategy]
    collection = get_or_create_collection(collection_name)

    current_count = collection.count()
    if current_count > 0:
        return current_count

    chunks = load_chunks(CHUNK_FILES[strategy])
    index_chunks(chunks, collection_name=collection_name)

    return collection.count()


def search(
    query: str, top_k: int = 5, strategy: str = "section"
) -> list[dict[str, Any]]:
    """
    Search the selected collection and return structured retrieval results.
    """
    collection_name = COLLECTION_NAMES[strategy]
    collection = get_or_create_collection(collection_name)

    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    ids = results.get("ids", [[]])[0]
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    structured_results: list[dict[str, Any]] = []
    for chunk_id, document, metadata, distance in zip(
        ids, documents, metadatas, distances
    ):
        structured_results.append(
            {
                "chunk_id": chunk_id,
                "text": document,
                "score": distance,  # Chroma distance; lower is generally better
                "pmid": metadata.get("pmid", ""),
                "title": metadata.get("title", ""),
                "section": metadata.get("section", ""),
                "chunk_index": metadata.get("chunk_index", -1),
                "strategy": metadata.get("strategy", strategy),
                "topic": metadata.get("topic", ""),
            }
        )

    return structured_results


def build_index_from_file(strategy: str) -> None:
    """Index all chunks for the requested strategy."""
    chunk_file = CHUNK_FILES[strategy]
    collection_name = COLLECTION_NAMES[strategy]

    chunks = load_chunks(chunk_file)
    print(f"Loaded {len(chunks)} chunks from {chunk_file}")

    index_chunks(chunks, collection_name=collection_name)
    print(f"Indexed into Chroma collection: {collection_name}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["fixed", "section"], default="section")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    build_index_from_file(strategy=args.strategy)


if __name__ == "__main__":
    main()
