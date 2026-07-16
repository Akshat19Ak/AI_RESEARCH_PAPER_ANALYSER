"""
chains.py — Analysis chains that connect retrieval → prompting → LLM.

Each function in this module:
    1. Retrieves relevant chunks using hybrid search + reranking
    2. Formats them into a prompt template
    3. Sends the prompt to the LLM
    4. Returns the answer + source documents (for citations)

This is the "brain" of the RAG pipeline — where retrieval meets generation.
"""

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from backend.src.retrieval.bm25_store import BM25Store
from backend.src.retrieval.hybrid_retriever import hybrid_search
from backend.src.retrieval.reranker import rerank
from backend.src.utils.helpers import format_context, format_history
from backend.src.generation.prompts import (
    STRUCTURED_SUMMARY_PROMPT,
    CHAT_PROMPT,
    QUICK_INSIGHTS_PROMPT,
    SECTION_DIVE_PROMPT,
    INTERVIEW_PROMPT,
    COMPARE_PROMPT,
)


# ══════════════════════════════════════════════════════════════════════════════
#  CORE RETRIEVAL HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _retrieve(
    vectorstore: FAISS,
    bm25_store: BM25Store,
    query: str,
    top_k: int = 5,
) -> list[Document]:
    """
    Full retrieval pipeline: Hybrid Search → Reranking.

    This is the core retrieval used by ALL analysis chains.
    It combines BM25 + FAISS results, then reranks with cross-encoder.

    Args:
        vectorstore: FAISS index.
        bm25_store: BM25 index.
        query: Search query.
        top_k: Final number of chunks to return after reranking.

    Returns:
        List of the most relevant Document chunks.
    """
    # Step 1: Hybrid search (BM25 + FAISS with Reciprocal Rank Fusion)
    candidates = hybrid_search(vectorstore, bm25_store, query)

    # Step 2: Rerank with cross-encoder for precision
    return rerank(query, candidates, top_k=top_k)


# ══════════════════════════════════════════════════════════════════════════════
#  CHAIN 1: STRUCTURED SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def run_structured_summary(llm, vectorstore: FAISS, bm25_store: BM25Store) -> tuple[str, list[Document]]:
    """
    Generate a full structured paper summary.

    Strategy: Use 5 DIFFERENT queries to retrieve a BROAD range of chunks,
    covering all major sections of the paper. This prevents the summary
    from being biased toward just one section.

    Returns:
        Tuple of (summary_text, source_documents).
    """
    # ── Broad retrieval: 5 targeted queries for full paper coverage ─────
    breadth_queries = [
        "problem statement motivation introduction",
        "methodology approach model architecture algorithm",
        "experiments results performance benchmarks",
        "contributions novel innovation",
        "limitations future work conclusion",
    ]

    all_docs = []
    seen_chunks = set()

    for q in breadth_queries:
        docs = _retrieve(vectorstore, bm25_store, q, top_k=3)
        for d in docs:
            # Deduplicate chunks across queries
            key = d.metadata.get("chunk_index", d.page_content[:50])
            if key not in seen_chunks:
                all_docs.append(d)
                seen_chunks.add(key)

    # ── Format and generate ─────────────────────────────────────────────
    context = format_context(all_docs[:12])  # Cap at 12 chunks
    prompt = STRUCTURED_SUMMARY_PROMPT.format(context=context)
    response = llm.invoke(prompt)

    return response.content, all_docs[:12]


# ══════════════════════════════════════════════════════════════════════════════
#  CHAIN 2: CHAT Q&A
# ══════════════════════════════════════════════════════════════════════════════

def run_chat(
    llm,
    vectorstore: FAISS,
    bm25_store: BM25Store,
    question: str,
    history: list,
    difficulty: str = "Expert",
) -> tuple[str, list[Document], list]:
    """
    Answer a free-form question with conversation history and citations.

    Args:
        llm: Groq LLM instance.
        vectorstore: FAISS index.
        bm25_store: BM25 index.
        question: User's question.
        history: List of previous chat messages.
        difficulty: "Beginner" or "Expert" — adjusts explanation depth.

    Returns:
        Tuple of (answer_text, source_documents, updated_history).
    """
    # ── Retrieve relevant chunks ────────────────────────────────────────
    docs = _retrieve(vectorstore, bm25_store, question, top_k=5)

    # ── Build prompt with context + history + difficulty ─────────────────
    context = format_context(docs)
    hist_str = format_history(history)
    prompt = CHAT_PROMPT.format(
        context=context,
        history=hist_str,
        question=question,
        difficulty=difficulty,
    )

    response = llm.invoke(prompt)
    answer = response.content

    # ── Update conversation history ─────────────────────────────────────
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})

    return answer, docs, history


# ══════════════════════════════════════════════════════════════════════════════
#  CHAIN 3: QUICK INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════

def run_quick_insights(llm, vectorstore: FAISS, bm25_store: BM25Store) -> tuple[str, list[Document]]:
    """
    Extract the 5 most important insights from the paper.

    Returns:
        Tuple of (insights_text, source_documents).
    """
    docs = _retrieve(
        vectorstore, bm25_store,
        "main findings results key contributions innovations",
        top_k=5,
    )
    context = format_context(docs)
    prompt = QUICK_INSIGHTS_PROMPT.format(context=context)
    response = llm.invoke(prompt)
    return response.content, docs


# ══════════════════════════════════════════════════════════════════════════════
#  CHAIN 4: SECTION DEEP-DIVE
# ══════════════════════════════════════════════════════════════════════════════

def run_section_dive(
    llm,
    vectorstore: FAISS,
    bm25_store: BM25Store,
) -> tuple[str, list[Document]]:
    """
    Deep-dive into the complete research paper.

    Returns:
        Tuple of (analysis_text, source_documents).
    """
    queries = ["abstract background problem", "methodology architecture approach", "experiments results conclusion"]
    all_docs = []
    seen = set()
    for q in queries:
        for d in _retrieve(vectorstore, bm25_store, q, top_k=4):
            key = d.metadata.get("chunk_index", d.page_content[:50])
            if key not in seen:
                all_docs.append(d)
                seen.add(key)
    context = format_context(all_docs[:12])
    prompt = SECTION_DIVE_PROMPT.format(context=context)
    response = llm.invoke(prompt)
    return response.content, all_docs[:12]


# ══════════════════════════════════════════════════════════════════════════════
#  CHAIN 5: INTERVIEW QUESTION GENERATION (WOW FEATURE)
# ══════════════════════════════════════════════════════════════════════════════

def run_interview_prep(llm, vectorstore: FAISS, bm25_store: BM25Store) -> tuple[str, list[Document]]:
    """
    Generate interview questions (Easy/Medium/Hard) based on the paper.

    This is a WOW FEATURE — very few RAG projects include this.
    It demonstrates practical application of the RAG pipeline
    beyond simple Q&A.

    Returns:
        Tuple of (questions_text, source_documents).
    """
    # Retrieve a broad context (similar to structured summary)
    queries = [
        "core methodology and approach",
        "results performance and benchmarks",
        "problem statement and motivation",
    ]
    all_docs = []
    seen = set()
    for q in queries:
        docs = _retrieve(vectorstore, bm25_store, q, top_k=3)
        for d in docs:
            key = d.metadata.get("chunk_index", d.page_content[:50])
            if key not in seen:
                all_docs.append(d)
                seen.add(key)

    context = format_context(all_docs[:10])
    prompt = INTERVIEW_PROMPT.format(context=context)
    response = llm.invoke(prompt)
    return response.content, all_docs[:10]


# ══════════════════════════════════════════════════════════════════════════════
#  CHAIN 6: PAPER COMPARISON (WOW FEATURE)
# ══════════════════════════════════════════════════════════════════════════════

def run_comparison(
    llm,
    vectorstore: FAISS,
    bm25_store: BM25Store,
    paper1_summary: str,
) -> str:
    """
    Compare the current paper with a previously analyzed paper.

    Strategy:
        1. The user generates a summary for Paper 1 (stored in session)
        2. They load Paper 2 and click "Compare"
        3. We retrieve broad context from Paper 2
        4. Send Paper 1's summary + Paper 2's context to the LLM

    This avoids storing two vector stores simultaneously (saves memory).

    Args:
        llm: Groq LLM instance.
        vectorstore: FAISS index of the CURRENT (Paper 2) document.
        bm25_store: BM25 index of current document.
        paper1_summary: Stored structured summary of the previous paper.

    Returns:
        Comparison analysis text.
    """
    # Get broad context from current paper
    docs = _retrieve(
        vectorstore, bm25_store,
        "methodology results contributions problem",
        top_k=6,
    )
    paper2_context = format_context(docs)

    prompt = COMPARE_PROMPT.format(
        paper1_summary=paper1_summary,
        paper2_context=paper2_context,
    )
    response = llm.invoke(prompt)
    return response.content
