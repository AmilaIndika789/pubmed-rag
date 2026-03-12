"""
Minimal Streamlit app for medical RAG.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import os
from typing import Any

import streamlit as st
from streamlit import errors

from rag.pipeline import answer_question
from utils import load_env
from vectordb.store import ensure_collection_ready


def load_secrets() -> None:
    """
    Safely load secrets from Streamlit or fall back to environment variables.
    """
    try:
        # Accessing st.secrets directly to trigger the check
        for key in ["GEMINI_API_KEY", "HF_TOKEN", "NCBI_API_KEY"]:
            if key in st.secrets:
                os.environ[key] = st.secrets[key]
    except (FileNotFoundError, errors.StreamlitSecretNotFoundError) as e:
        print(f"Secrets file not found: {e}")


@st.cache_resource
def initialize_runtime_resources() -> dict[str, int]:
    """
    Initialize resources needed by the app.

    This runs once per app process and is reused across reruns.
    It ensures both vector collections are available before queries.
    """
    load_env()
    load_secrets()

    section_count = ensure_collection_ready("section")
    fixed_count = ensure_collection_ready("fixed")

    return {
        "section": section_count,
        "fixed": fixed_count,
    }


def render_sources(sources: list[dict[str, Any]]) -> None:
    """Render deduplicated sources."""
    if not sources:
        st.write("No sources available.")
        return

    for source in sources:
        st.markdown(f"- **PMID {source['pmid']}** — {source['title']}")


def render_chunks(chunks: list[dict[str, Any]]) -> None:
    """Render retrieved chunks in collapsible sections."""
    for idx, chunk in enumerate(chunks, start=1):
        with st.expander(f"Chunk {idx} | PMID {chunk['pmid']} | {chunk['section']}"):
            st.write(f"**Title:** {chunk['title']}")
            st.write(f"**Score:** {chunk['score']}")
            st.write(chunk["text"])


def main() -> None:
    """Streamlit app entrypoint."""
    st.set_page_config(page_title="Medical Literature RAG Demo", layout="wide")

    status_placeholder = st.empty()

    st.title("Medical Literature RAG Demo")
    st.caption("PubMed + ChromaDB + Gemini")
    st.warning("For demo purposes only. Not medical advice.")

    try:
        status_placeholder.warning("System status: Loading ... Please wait!!")
        initialize_runtime_resources()
        status_placeholder.success("System status: Up and running. Ask your question.")
    except Exception as exc:
        status_placeholder.error(f"System status: Failed to initialize. {exc}")
        st.stop()

    question = st.text_input("Enter a medical question")
    strategy = st.selectbox("Chunking strategy", ["section", "fixed"], index=0)
    top_k = st.slider("Top-k retrieval", min_value=3, max_value=8, value=5, step=1)
    prompt_version = st.selectbox("Prompt version", ["v2", "v1"], index=0)

    if st.button("Search"):
        if not question.strip():
            st.error("Please enter a question.")
            return

        with st.spinner("Retrieving articles and generating answer..."):
            result = answer_question(
                question=question,
                top_k=top_k,
                strategy=strategy,
                prompt_version=prompt_version,
            )

        st.subheader("Answer")
        st.write(result["answer"])

        st.subheader("Sources")
        render_sources(result["sources"])

        st.subheader("Retrieved Chunks")
        render_chunks(result["used_context"])


if __name__ == "__main__":
    main()
