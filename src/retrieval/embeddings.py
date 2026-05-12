"""
embeddings.py — Embedding model management.

We use HuggingFace's MiniLM-L6-v2 for all embedding operations:
    - Document chunk embedding (indexing phase)
    - Query embedding (search phase)
    - Evaluation metric computation (scoring phase)

WHY MiniLM-L6-v2:
    - Only 23MB model size (tiny)
    - Runs locally on CPU — zero API cost, works offline
    - 384-dimensional output vectors
    - Surprisingly good quality for semantic similarity tasks
    - Used in production by many RAG systems
"""

import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.utils.config import EMBEDDING_MODEL, EMBEDDING_DEVICE


@st.cache_resource(show_spinner=False)
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Load and cache the embedding model (singleton pattern).

    @st.cache_resource ensures the model is loaded ONCE and reused
    across all Streamlit reruns. Without this, the 23MB model would
    reload on every button click (~3 seconds wasted each time).

    Returns:
        HuggingFaceEmbeddings instance ready for encoding.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
        # normalize_embeddings=True makes cosine similarity = dot product,
        # which is faster to compute and gives identical ranking results.
    )
