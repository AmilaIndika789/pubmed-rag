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
import xml.etree.ElementTree as ET

import xml.dom.minidom

from utils import load_env

PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

OUTPUT_PATH = Path("ingest/data/JSON/pubmed_articles.json")

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
    return payload.get("esearchresult", {}).get("idlist", [])


def fetch_pubmed_xml(pmids: list[str]) -> str:
    """
    Fetch PubMed records as XML for a list of PMIDs.

    Args:
        pmids: List of PubMed IDs.

    Returns:
        Raw XML string.
    """
    if not pmids:
        return ""

    load_env()
    ncbi_api_key = os.getenv("NCBI_API_KEY")

    params: dict[str, str] = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "api_key": ncbi_api_key,
    }

    response = safe_get(PUBMED_EFETCH_URL, params=params)
    return response.text


def extract_text(element: ET.Element | None) -> str:
    """Safely extract all text from an XML element."""
    if element is None:
        return ""
    return "".join(element.itertext()).strip()


def parse_authors(article_elem: ET.Element) -> list[str]:
    """
    Parse author names from a PubMed article element.

    Returns:
        List like ['Jane Doe', 'John Smith'].
    """
    authors: list[str] = []
    for author in article_elem.findall(".//Author"):
        last_name = extract_text(author.find("LastName"))
        fore_name = extract_text(author.find("ForeName"))
        collective_name = extract_text(author.find("CollectiveName"))

        if collective_name:
            authors.append(collective_name)
        elif last_name or fore_name:
            authors.append(f"{fore_name} {last_name}".strip())

    return authors


def parse_mesh_terms(article_elem: ET.Element) -> list[str]:
    """Extract MeSH descriptor names."""
    mesh_terms: list[str] = []
    for mesh in article_elem.findall(".//MeshHeading"):
        descriptor = extract_text(mesh.find("DescriptorName"))
        if descriptor:
            mesh_terms.append(descriptor)
    return mesh_terms


def parse_publication_date(article_elem: ET.Element) -> str:
    """
    Try to construct a simple publication date string.

    Returns:
        YYYY-MM-DD when possible, else partial like YYYY-MM, else YYYY, else ''.
    """
    pub_date = article_elem.find(".//PubDate")
    if pub_date is None:
        return ""

    year = extract_text(pub_date.find("Year"))
    month = extract_text(pub_date.find("Month"))
    day = extract_text(pub_date.find("Day"))

    # TODO: Normalize month names if needed.
    if year and month and day:
        return f"{year}-{month}-{day}"
    if year and month:
        return f"{year}-{month}"
    return year


def parse_abstract(article_elem: ET.Element) -> str:
    """
    Extract abstract text.

    If there are multiple AbstractText nodes, join them with blank lines.
    """
    parts: list[str] = []
    for abstract_text in article_elem.findall(".//Abstract/AbstractText"):
        label = abstract_text.attrib.get("Label", "").strip()
        text = extract_text(abstract_text)
        if not text:
            continue

        if label:
            parts.append(f"{label}: {text}")
        else:
            parts.append(text)

    return "\n\n".join(parts).strip()


def parse_single_article(
    pubmed_article_elem: ET.Element, topic: str
) -> dict[str, Any] | None:
    """
    Parse one PubMedArticle XML element into a structured dictionary.

    Returns:
        Structured record, or None if critical fields are missing.
    """
    pmid = extract_text(pubmed_article_elem.find(".//PMID"))
    title = extract_text(pubmed_article_elem.find(".//ArticleTitle"))
    abstract = parse_abstract(pubmed_article_elem)
    authors = parse_authors(pubmed_article_elem)
    publication_date = parse_publication_date(pubmed_article_elem)
    mesh_terms = parse_mesh_terms(pubmed_article_elem)

    if not pmid or not title or not abstract:
        return None

    return {
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "publication_date": publication_date,
        "mesh_terms": mesh_terms,
        "topic": topic,
    }


def main() -> None:
    """
    Main ingestion pipeline.
    """
    topic = "diabetes"
    query = TOPICS[topic]
    print(f"Fetching topic={topic} | query={query}")
    pmids = search_pubmed(query=query)
    pubmed_xml = fetch_pubmed_xml(pmids)

    with open(f"ingest/data/XML/{topic}_pubmed.xml", "w", encoding="utf-8") as f:
        f.write(pubmed_xml)

    root = ET.fromstring(pubmed_xml)
    article = root.find(".//PubmedArticle")
    print(json.dumps(parse_single_article(article, topic), indent=4))


if __name__ == "__main__":
    main()
