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


def main() -> None:
    pass


if __name__ == "__main__":
    main()
