"""
reranker.py — Cross-encoder reranking for precision retrieval.

WHY RERANKING MATTERS:
    Initial retrieval (BM25 + FAISS) is fast but approximate.
    A cross-encoder RERANKER scores each (query, chunk) pair together,
    achieving much higher accuracy than bi-encoder similarity.

HOW IT WORKS:
    Bi-encoder (FAISS):    Embeds query and chunks SEPARATELY → fast but less precise
    Cross-encoder (Reranker): Processes query + chunk TOGETHER → slow but very precise

    We use bi-encoder for initial retrieval (fast, top-10),
    then cross-encoder to re-score and pick the best 5.

MODEL: cross-encoder/ms-marco-MiniLM-L-6-v2
    - Only ~23MB (very small)
    - Trained on MS MARCO (the gold standard for search relevance)
    - Runs on CPU in milliseconds per pair
    - Part of the sentence-transformers library (already installed)
"""

import functools
from langchain_core.documents import Document
from backend.src.utils.config import RERANKER_MODEL, FINAL_TOP_K


@functools.lru_cache(maxsize=1)
def get_reranker():
    """
    Load and cache the cross-encoder reranker model (lazy singleton).

    The model is loaded ONLY when reranking is first triggered,
    not at app startup. This saves ~23MB of RAM until needed.

    Returns:
        CrossEncoder model instance.
    """
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANKER_MODEL)


def rerank(query: str, docs: list[Document], top_k: int = FINAL_TOP_K) -> list[Document]:
    """
    Re-score and re-rank retrieved documents using cross-encoder.

    Pipeline:
        1. Take the initial retrieval results (e.g., 10 chunks)
        2. Score each (query, chunk) pair with the cross-encoder
        3. Sort by cross-encoder score (descending)
        4. Return only the top-k most relevant chunks

    This step typically improves answer quality by 15-30% compared
    to using raw retrieval results.

    Args:
        query: The user's question.
        docs: List of Document chunks from hybrid retrieval.
        top_k: How many chunks to keep after reranking.

    Returns:
        List of top_k Document chunks, ordered by relevance.
    """
    if not docs:
        return []

    # ── Load the cross-encoder model (cached after first use) ───────────
    reranker = get_reranker()

    # ── Create (query, chunk) pairs for scoring ─────────────────────────
    pairs = [(query, doc.page_content) for doc in docs]

    # ── Score all pairs ─────────────────────────────────────────────────
    # The cross-encoder processes query and document TOGETHER through
    # all transformer layers, giving a much more accurate relevance score
    # than comparing independent embeddings.
    scores = reranker.predict(pairs)

    # ── Sort by score and return top-k ──────────────────────────────────
    scored_docs = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True,
    )

    # Attach reranker score to metadata (useful for evaluation & display)
    results = []
    for score, doc in scored_docs[:top_k]:
        doc.metadata["reranker_score"] = float(score)
        results.append(doc)

    return results
