"""
metrics.py — Evaluation metrics for RAG answer quality.

ALL METRICS ARE FREE — they use the local MiniLM embedding model only.
No OpenAI API key needed. No external service calls.

METRICS IMPLEMENTED:
    1. Answer Relevance  → How well the answer addresses the question
    2. Faithfulness       → How much of the answer is grounded in source chunks
    3. Context Precision  → How relevant the retrieved chunks are to the query
    4. Overall Score      → Weighted average of all metrics

HOW THEY WORK:
    We use COSINE SIMILARITY between embeddings to measure semantic closeness.
    - Embed the question, answer, and source chunks with MiniLM
    - Compare them pairwise to compute each metric
    - All computation is local (CPU), takes ~100ms total

WHY THIS MATTERS FOR YOUR RESUME:
    Evaluation metrics are RARE in student RAG projects.
    Having them shows you understand production ML quality assurance.
"""

import numpy as np
from langchain_core.documents import Document
from src.retrieval.embeddings import get_embeddings
from src.utils.config import FAITHFULNESS_THRESHOLD


def compute_all_metrics(
    question: str,
    answer: str,
    source_docs: list[Document],
) -> dict:
    """
    Compute all RAG evaluation metrics for a single Q&A interaction.

    This is the main entry point — call this after generating an answer.

    Args:
        question: The user's original question.
        answer: The LLM-generated answer.
        source_docs: The retrieved document chunks used as context.

    Returns:
        Dictionary with all metric scores (0-100 scale) and metadata:
        {
            "answer_relevance": 85.2,
            "faithfulness": 91.0,
            "context_precision": 78.5,
            "overall_score": 84.9,
            "faithful_sentences": 9,
            "total_sentences": 10,
            "details": {...}
        }
    """
    embeddings_model = get_embeddings()

    # ── Embed everything we need ────────────────────────────────────────
    # We batch all embeddings into a single call for efficiency.
    texts_to_embed = [question, answer]
    chunk_texts = [doc.page_content for doc in source_docs]
    texts_to_embed.extend(chunk_texts)

    all_embeddings = embeddings_model.embed_documents(texts_to_embed)

    question_emb = np.array(all_embeddings[0])
    answer_emb = np.array(all_embeddings[1])
    chunk_embs = np.array(all_embeddings[2:])

    # ── Compute individual metrics ──────────────────────────────────────
    relevance = _answer_relevance(question_emb, answer_emb)
    faith_score, faith_count, total_sents = _faithfulness(
        answer, chunk_texts, embeddings_model
    )
    ctx_precision = _context_precision(question_emb, chunk_embs)

    # ── Overall score (weighted average) ────────────────────────────────
    # Weights: Faithfulness matters most (hallucination prevention),
    # then relevance, then context precision.
    overall = (faith_score * 0.4 + relevance * 0.35 + ctx_precision * 0.25)

    return {
        "answer_relevance": round(relevance, 1),
        "faithfulness": round(faith_score, 1),
        "context_precision": round(ctx_precision, 1),
        "overall_score": round(overall, 1),
        "faithful_sentences": faith_count,
        "total_sentences": total_sents,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  METRIC 1: ANSWER RELEVANCE
# ══════════════════════════════════════════════════════════════════════════════

def _answer_relevance(question_emb: np.ndarray, answer_emb: np.ndarray) -> float:
    """
    Measure how well the answer addresses the question.

    HOW: Cosine similarity between the question and answer embeddings.

    High score = the answer is semantically about the same topic as the question.
    Low score  = the answer is off-topic or too generic.

    Example:
        Q: "What optimizer was used?"
        A: "The paper uses AdamW with lr=3e-4" → HIGH relevance
        A: "The paper was published in 2023"   → LOW relevance

    Returns:
        Score from 0 to 100.
    """
    # Cosine similarity of normalized vectors = dot product
    similarity = np.dot(question_emb, answer_emb)
    # Clamp to [0, 1] and scale to percentage
    return float(max(0, min(1, similarity))) * 100


# ══════════════════════════════════════════════════════════════════════════════
#  METRIC 2: FAITHFULNESS (Anti-Hallucination)
# ══════════════════════════════════════════════════════════════════════════════

def _faithfulness(
    answer: str,
    chunk_texts: list[str],
    embeddings_model,
) -> tuple[float, int, int]:
    """
    Measure how much of the answer is grounded in the source documents.

    HOW:
        1. Split the answer into individual sentences
        2. Embed each sentence
        3. For each sentence, find its maximum similarity to ANY source chunk
        4. If max_similarity > threshold → sentence is "faithful"
        5. Faithfulness = % of faithful sentences

    WHY THIS WORKS:
        If the LLM hallucinated a fact, that sentence won't match any source chunk.
        The embedding similarity will be low, flagging it as potentially unfaithful.

    Returns:
        Tuple of (faithfulness_score_0_100, faithful_count, total_sentences).
    """
    # Split answer into sentences
    sentences = _split_sentences(answer)

    if not sentences or not chunk_texts:
        return 100.0, 0, 0  # No sentences to check

    # Embed all sentences
    sent_embeddings = embeddings_model.embed_documents(sentences)
    chunk_embeddings = embeddings_model.embed_documents(chunk_texts)

    sent_embs = np.array(sent_embeddings)
    chunk_embs = np.array(chunk_embeddings)

    # For each sentence, compute max similarity to any chunk
    faithful_count = 0
    for sent_emb in sent_embs:
        # Cosine similarity with all chunks
        similarities = np.dot(chunk_embs, sent_emb)
        max_sim = np.max(similarities)

        if max_sim >= FAITHFULNESS_THRESHOLD:
            faithful_count += 1

    total = len(sentences)
    score = (faithful_count / total * 100) if total > 0 else 100.0
    return score, faithful_count, total


# ══════════════════════════════════════════════════════════════════════════════
#  METRIC 3: CONTEXT PRECISION
# ══════════════════════════════════════════════════════════════════════════════

def _context_precision(question_emb: np.ndarray, chunk_embs: np.ndarray) -> float:
    """
    Measure how relevant the retrieved chunks are to the question.

    HOW: Average cosine similarity between the question and each chunk.

    High score = retrieved chunks are all relevant (good retrieval).
    Low score  = some retrieved chunks are noise (retrieval could improve).

    This metric tells you whether the RETRIEVAL step is working well,
    independent of how good the LLM answer is.

    Returns:
        Score from 0 to 100.
    """
    if len(chunk_embs) == 0:
        return 0.0

    similarities = np.dot(chunk_embs, question_emb)
    avg_sim = float(np.mean(similarities))
    return max(0, min(1, avg_sim)) * 100


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: SENTENCE SPLITTING
# ══════════════════════════════════════════════════════════════════════════════

def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences for faithfulness checking.

    Uses simple regex-based splitting (handles ., !, ? followed by space).
    Filters out very short fragments (< 10 chars) that aren't real sentences.
    """
    import re
    # Split on sentence-ending punctuation followed by space or newline
    raw = re.split(r'(?<=[.!?])\s+', text)
    # Filter out very short fragments and list markers
    return [s.strip() for s in raw if len(s.strip()) > 10]
