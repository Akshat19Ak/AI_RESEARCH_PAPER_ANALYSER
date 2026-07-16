"""
vector_store.py — FAISS vector store for dense (semantic) retrieval.

FAISS (Facebook AI Similarity Search) stores document chunk embeddings
and finds the most semantically similar chunks to a query in milliseconds.

HOW IT WORKS:
    1. Each chunk is embedded into a 384-dim vector by MiniLM-L6-v2
    2. FAISS indexes all vectors using Inner Product search
    3. When you ask a question, your question is embedded too
    4. FAISS finds the k vectors closest to your question vector
    5. The corresponding text chunks are returned as context

WHY FAISS (not Pinecone, ChromaDB, etc.):
    - In-memory = sub-millisecond search on a laptop
    - No external service needed — works fully offline
    - Perfect for single-document RAG sessions
    - Used in production by Meta, Spotify, and others
"""

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from backend.src.retrieval.embeddings import get_embeddings


def build_vector_store(docs: list[Document]) -> FAISS:
    """
    Build a FAISS index from document chunks.

    This is the INDEXING phase — it runs once per document.
    Each chunk is embedded (converted to a 384-dim vector) and
    stored in the FAISS index for fast similarity search.

    Args:
        docs: List of LangChain Document objects from the chunker.

    Returns:
        FAISS vector store ready for similarity search.
    """
    import gc

    embeddings = get_embeddings()
    
    # Process in batches to keep peak RAM low
    batch_size = 50
    vector_store = None
    
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i : i + batch_size]
        if vector_store is None:
            vector_store = FAISS.from_documents(batch_docs, embeddings)
        else:
            vector_store.add_documents(batch_docs)
            
        # Force garbage collection of intermediate tensors
        gc.collect()

    return vector_store


def dense_search(vectorstore: FAISS, query: str, k: int = 10) -> list[Document]:
    """
    Retrieve the top-k most semantically similar chunks to the query.

    This is DENSE retrieval — it finds chunks whose MEANING is similar
    to the question, even if they don't share the same exact words.

    Example:
        Query: "How does the model handle long sequences?"
        Finds: chunk about "processing extended input lengths" (same meaning, different words)

    Args:
        vectorstore: The FAISS index built from document chunks.
        query: User's question or search query.
        k: Number of top results to return.

    Returns:
        List of the k most relevant Document chunks.
    """
    return vectorstore.similarity_search(query, k=k)
