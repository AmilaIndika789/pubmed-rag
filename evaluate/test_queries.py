"""
Basic retrieval sanity-check script.

Run:
    python -m evaluate.test_queries
"""

from __future__ import annotations

import inspect
from typing import Any

from vectordb.store import search

TEST_QUERIES = [
    "What are the current first-line treatments for type 2 diabetes?",
    "What are the side effects of ACE inhibitors in hypertension?",
    "How effective are inhaled corticosteroids for childhood asthma?",
    "What is a completely unrelated topic like quantum computing?",
    "Compare metformin and sulfonylureas for glycemic control",
]


def print_retrieval_results(query: str, results: list[dict[str, Any]]) -> None:
    """Pretty-print retrieval results."""
    header = f"{'='*100}\nQUERY: {query}\n{'-'*100}"
    print(header)

    for idx, item in enumerate(results, start=1):
        # Clean up the text preview for single-line display
        preview = str(item.get("text", ""))[:250].replace("\n", " ")
        output = inspect.cleandoc(
            f"""
            [{idx}] Title: {item.get('title', 'N/A')}
                PMID: {item.get('pmid', 'N/A')}
                Section: {item.get('section', 'N/A')}
                Score: {item.get('score', 0.0):.4f}
                Preview: {preview}...
            """
        )
        print(f"{output}\n")


def run_retrieval_eval(strategy: str = "section", top_k: int = 5) -> None:
    """Run retrieval-only evaluation."""
    for query in TEST_QUERIES:
        results = search(query=query, top_k=top_k, strategy=strategy)
        print_retrieval_results(query, results)


def main() -> None:
    """
    Retrieval eval - to test retrieval from the ChromaDB
    """
    run_retrieval_eval(strategy="section", top_k=5)


if __name__ == "__main__":
    main()
