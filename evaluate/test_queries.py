"""
Basic retrieval sanity-check script.

Run:
    python -m evaluate.test_queries
"""

from __future__ import annotations

from typing import Any

from vectordb.store import search
from rag.pipeline import answer_question

TEST_QUERIES = [
    "What are the current first-line treatments for type 2 diabetes?",
    "What are the side effects of ACE inhibitors in hypertension?",
    "How effective are inhaled corticosteroids for childhood asthma?",
    "What is a completely unrelated topic like quantum computing?",
    "Compare metformin and sulfonylureas for glycemic control",
]


def print_retrieval_results(query: str, results: list[dict[str, Any]]) -> None:
    """Pretty-print retrieval results."""
    print("=" * 100)
    print(f"QUERY: {query}")
    print("-" * 100)

    for idx, item in enumerate(results, start=1):
        preview = item["text"][:250].replace("\n", " ")
        print(f"[{idx}] Title: {item['title']}")
        print(f"    PMID: {item['pmid']}")
        print(f"    Section: {item['section']}")
        print(f"    Score: {item['score']}")
        print(f"    Preview: {preview}...")
        print()


def run_retrieval_eval(strategy: str = "section", top_k: int = 5) -> None:
    """Run retrieval-only evaluation."""
    for query in TEST_QUERIES:
        results = search(query=query, top_k=top_k, strategy=strategy)
        print_retrieval_results(query, results)


def run_generation_eval(strategy: str = "section", top_k: int = 5) -> None:
    """Run full RAG generation evaluation."""
    for query in TEST_QUERIES:
        result = answer_question(question=query, top_k=top_k, strategy=strategy)
        print("=" * 100)
        print(f"QUERY: {query}")
        print("-" * 100)
        print("ANSWER:")
        print(result["answer"])
        print("\nSOURCES:")
        for source in result["sources"]:
            print(f"  PMID {source['pmid']} | {source['title']}")
        print()


def main() -> None:
    """
    Retrieval eval - to test retrieval from the ChromaDB
    Generation eval - to test Gemini LLM responses
    """
    # run_retrieval_eval(strategy="section", top_k=5)
    run_generation_eval(strategy="section", top_k=5)


if __name__ == "__main__":
    main()
