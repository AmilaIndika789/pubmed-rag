# PubMed Medical RAG Demo

A minimal end-to-end medical Retrieval-Augmented Generation (RAG) system that helps healthcare professionals find accurate, evidence-based answers from medical literature. It ingests PubMed articles, chunks them using two strategies, embeds and stores them in ChromaDB, retrieves relevant context for a medical question, and generates grounded answers with Gemini LLM through a simple Streamlit UI.

## What this project does

This project answers medical literature questions using only retrieved PubMed article context. The pipeline:

1. fetches article metadata and abstracts from PubMed using NCBI E-utilities,
2. stores the raw data in structured JSON,
3. creates chunks using both fixed-size and section-based strategies,
4. embeds chunks with `sentence-transformers/all-MiniLM-L6-v2`,
5. stores them in ChromaDB with metadata,
6. retrieves top-k relevant chunks for a user query,
7. prompts Gemini to answer using only the retrieved context and cite PMIDs,
8. returns the answer and sources in a Streamlit app.

## Tech stack

- Python 3.10+
- PubMed / NCBI E-utilities for ingestion
- HuggingFace `sentence-transformers/all-MiniLM-L6-v2` for embeddings
- ChromaDB for vector storage and retrieval
- Gemini LLM/s via `google-genai` for generation
- Streamlit for the demo UI

## Repository structure

```text
pubmed-rag/
├── README.md
├── LICENSE
├── requirements.txt
├── .env
├── ingest/
│   ├── pubmed_fetcher.py
│   └── data/
├── chunking/
│   ├── chunker.py
│   └── comparison.md
├── vectordb/
│   ├── embeddings.py
│   ├── store.py
│   └── db/
├── rag/
│   ├── pipeline.py
│   └── prompts.py
├── evaluate/
│   └── test_queries.py
└── app.py
└── utils.py
```
