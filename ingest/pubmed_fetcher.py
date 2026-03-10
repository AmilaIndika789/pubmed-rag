"""
Fetch PubMed articles for a small set of medical topics and save structured JSON.

Run:
    python ingest/pubmed_fetcher.py
"""

from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import Any
import requests

from utils import load_env

PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

OUTPUT_PATH = Path("ingest/data/pubmed_articles.json")

TOPICS = {
    "diabetes": "type 2 diabetes treatment guideline",
    "hypertension": "ACE inhibitors hypertension side effects guideline",
    "asthma": "childhood asthma inhaled corticosteroids",
}


def ensure_output_dir(path: Path) -> None:
    """Create parent directories if they do not already exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


def safe_get(
    url: str, params: dict[str, Any], timeout: int = 30, retries: int = 3
) -> requests.Response:
    """
    Make a GET request with very simple retry handling.

    Raises:
        RuntimeError: if all retries fail.
    """
    last_error: Exception | None = None

    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response
        except Exception as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"Request failed after {retries} retries: {last_error}")


def search_pubmed(query: str, retmax: int = 20) -> list[str]:
    """
    Search PubMed and return a list of PMIDs.

    Args:
        query: PubMed search query.
        retmax: Maximum number of PMIDs to return.
    """

    load_env()
    ncbi_api_key = os.getenv("NCBI_API_KEY")

    params: dict[str, str | int] = {
        "db": "pubmed",
        "term": query,
        "retmax": retmax,
        "retmode": "json",
        "sort": "relevance",
        "api_key": ncbi_api_key,
    }

    response = safe_get(PUBMED_ESEARCH_URL, params=params)
    payload = response.json()
    print(f"Payload: {json.dumps(payload, indent=4)}")
    return payload.get("esearchresult", {}).get("idlist", [])


def main() -> None:
    """
    Main ingestion pipeline.
    """
    topic = "diabetes"
    query = TOPICS[topic]
    print(f"Fetching topic={topic} | query={query}")
    pmids = search_pubmed(query=query)
    print(f"PMIDs: {pmids}")


if __name__ == "__main__":
    main()
