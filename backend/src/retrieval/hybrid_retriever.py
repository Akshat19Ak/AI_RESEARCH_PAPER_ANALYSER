"""
hybrid_retriever.py — Combines BM25 (keyword) + FAISS (semantic) retrieval.

HYBRID RETRIEVAL is the industry standard for production RAG systems.
It combines two complementary search strategies:

    BM25 (Sparse):  Finds chunks with matching KEYWORDS
    FAISS (Dense):   Finds chunks with matching MEANING

FUSION METHOD: Reciprocal Rank Fusion (RRF)
    Instead of comparing raw scores (which have different scales),
    RRF uses the RANK position from each retriever:

    RRF_score(doc) = 1/(k + rank_bm25) + 1/(k + rank_faiss)

    where k=60 is a constant that prevents high-ranked items
    from dominating. This is the same method used by major
    search engines and is more robust than score-based fusion.

PIPELINE:
    Query → BM25 (top-10) ─┐
                            ├→ Reciprocal Rank Fusion → Merged top-K
    Query → FAISS (top-10) ─┘
"""

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from backend.src.retrieval.bm25_store import BM25Store
import backend.src.retrieval.vector_store as vector_store_mod
from backend.src.utils.config import RETRIEVAL_TOP_K, HYBRID_ALPHA


def hybrid_search(
    vectorstore: FAISS,
    bm25_store: BM25Store,
    query: str,
    k: int = RETRIEVAL_TOP_K,
) -> list[Document]:
    """
    Perform hybrid retrieval using Reciprocal Rank Fusion.

    Steps:
        1. Get top-k results from FAISS (dense/semantic)
        2. Get top-k results from BM25 (sparse/keyword)
        3. Merge using Reciprocal Rank Fusion
        4. Return deduplicated, re-ranked results

    Args:
        vectorstore: FAISS vector store with document embeddings.
        bm25_store: BM25 index with keyword-tokenized documents.
        query: User's search query.
        k: Number of results to retrieve from each retriever.

    Returns:
        Merged list of Document chunks, ranked by combined relevance.
    """
    # ── Step 1: Get results from both retrievers ────────────────────────
    dense_results = vector_store_mod.dense_search(vectorstore, query, k=k)
    sparse_results = bm25_store.search(query, k=k)

    # ── Step 2: Apply Reciprocal Rank Fusion ────────────────────────────
    return reciprocal_rank_fusion(dense_results, sparse_results, k=k)


def reciprocal_rank_fusion(
    dense_docs: list[Document],
    sparse_docs: list[Document],
    k: int = 10,
    rrf_k: int = 60,
) -> list[Document]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion (RRF).

    RRF is superior to simple score averaging because:
    - BM25 and FAISS scores are on different scales (not comparable)
    - RRF only uses RANK positions, which are always comparable
    - It's the method used by Elasticsearch, Azure Cognitive Search, etc.

    Formula: RRF_score(doc) = Σ 1/(rrf_k + rank_i) for each retriever i

    Args:
        dense_docs: Ranked results from FAISS.
        sparse_docs: Ranked results from BM25.
        k: Number of final results to return.
        rrf_k: Smoothing constant (default 60, standard in literature).

    Returns:
        Merged and re-ranked list of Document chunks.
    """
    # Track RRF scores by document content (for deduplication)
    doc_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    # ── Score documents from DENSE (FAISS) retriever ────────────────────
    for rank, doc in enumerate(dense_docs):
        key = doc.page_content[:200]  # Use first 200 chars as dedup key
        score = 1.0 / (rrf_k + rank + 1)  # +1 because rank is 0-indexed
        doc_scores[key] = doc_scores.get(key, 0) + score
        doc_map[key] = doc

    # ── Score documents from SPARSE (BM25) retriever ────────────────────
    for rank, doc in enumerate(sparse_docs):
        key = doc.page_content[:200]
        score = 1.0 / (rrf_k + rank + 1)
        doc_scores[key] = doc_scores.get(key, 0) + score
        doc_map[key] = doc

    # ── Sort by combined RRF score and return top-k ─────────────────────
    sorted_keys = sorted(doc_scores, key=doc_scores.get, reverse=True)[:k]
    return [doc_map[key] for key in sorted_keys]
