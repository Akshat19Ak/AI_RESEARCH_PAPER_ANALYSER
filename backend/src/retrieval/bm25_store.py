"""
bm25_store.py — BM25 sparse retrieval for keyword-based search.

BM25 (Best Matching 25) is a KEYWORD-BASED search algorithm.
Unlike FAISS (which finds semantic similarity), BM25 finds chunks
that share the EXACT WORDS with your query.

WHY WE NEED BOTH BM25 AND FAISS:
    - FAISS excels at: "How does the model handle long sequences?"
      → Finds chunks about "processing extended input lengths"
    - BM25 excels at: "What is the BLEU score?"
      → Finds chunks containing the exact word "BLEU"

    Together, they form HYBRID RETRIEVAL — catching both semantic
    AND keyword matches. This is how production RAG systems work.

DEPENDENCY: rank_bm25 (pure Python, ~50KB — very lightweight)
"""

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


class BM25Store:
    """
    BM25 sparse retrieval index.

    Tokenizes document chunks into words and builds a BM25 index.
    At query time, returns chunks ranked by keyword relevance.
    """

    def __init__(self, docs: list[Document]):
        """
        Build BM25 index from document chunks.

        Args:
            docs: List of LangChain Document objects.
        """
        self.docs = docs

        # ── Tokenize each chunk into lowercase words ────────────────────
        # BM25 works on word tokens, not embeddings.
        tokenized = [doc.page_content.lower().split() for doc in docs]

        # ── Build BM25 index ────────────────────────────────────────────
        self.bm25 = BM25Okapi(tokenized)
        
        # Free the tokenized text list immediately to save RAM
        del tokenized

    def search(self, query: str, k: int = 10) -> list[Document]:
        """
        Retrieve top-k chunks by keyword relevance.

        Args:
            query: User's search query.
            k: Number of results to return.

        Returns:
            List of Document chunks ranked by BM25 score.
        """
        # Tokenize the query the same way we tokenized documents
        query_tokens = query.lower().split()

        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(query_tokens)

        # Sort by score (descending) and return top-k
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]

        return [self.docs[i] for i in ranked_indices]
