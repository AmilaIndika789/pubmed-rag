"""
Index chunks into ChromaDB and provide retrieval.

Run:
    python -m vectordb.store
"""

import chromadb
from chromadb.api.models.Collection import Collection

DB_DIR = "vectordb/db"


def get_chroma_client(persist_dir: str = DB_DIR) -> chromadb.PersistentClient:
    """Return a persistent Chroma client."""
    return chromadb.PersistentClient(path=persist_dir)


def get_or_create_collection(collection_name: str) -> Collection:
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


def main() -> None:
    pass


if __name__ == "__main__":
    main()
