"""
pinecone_store.py — Pinecone cloud vector store for scalable dense retrieval.

WHY PINECONE (over local FAISS):
    - Cloud-hosted = scales to millions of vectors
    - Serverless = no infrastructure management
    - Multi-user support = concurrent access
    - Production-ready = used by Notion, Shopify, etc.
    - Free tier = 100K vectors (enough for research papers)

ARCHITECTURE:
    - Each uploaded document gets its own NAMESPACE in the index
    - Namespaces provide logical isolation without extra indices
    - Metadata (chunk_index, source, page) stored alongside vectors
"""

import os
from langchain_core.documents import Document
from backend.src.retrieval.embeddings import get_embeddings


def get_pinecone_index(api_key: str = None, index_name: str = None):
    """
    Connect to or create a Pinecone index.

    Args:
        api_key: Pinecone API key (falls back to env var).
        index_name: Index name (falls back to env var or default).

    Returns:
        Pinecone Index object ready for upsert/query.
    """
    from pinecone import Pinecone, ServerlessSpec

    api_key = api_key or os.getenv("PINECONE_API_KEY", "")
    index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "rag-analyzer")

    if not api_key:
        raise ValueError("Pinecone API key not provided.")

    pc = Pinecone(api_key=api_key)

    # Create index if it doesn't exist (384 dims for MiniLM-L6-v2)
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    return pc.Index(index_name)


def upsert_to_pinecone(docs: list[Document], namespace: str, api_key: str = None):
    """
    Embed and upsert document chunks into Pinecone.

    Args:
        docs: LangChain Document chunks.
        namespace: Unique namespace for this document session.
        api_key: Pinecone API key.
    """
    index = get_pinecone_index(api_key)
    embed_model = get_embeddings()

    import gc
    batch_size = 50

    for batch_start in range(0, len(docs), batch_size):
        batch_docs = docs[batch_start : batch_start + batch_size]
        texts = [d.page_content for d in batch_docs]
        
        # Only embed this small batch
        embeddings = embed_model.embed_documents(texts)
        
        # Build upsert vectors
        vectors = []
        for i, (doc, emb) in enumerate(zip(batch_docs, embeddings)):
            global_index = batch_start + i
            vectors.append({
                "id": f"{namespace}-{global_index}",
                "values": emb,
                "metadata": {
                    "text": doc.page_content[:1000],  # Pinecone metadata limit
                    "chunk_index": doc.metadata.get("chunk_index", global_index),
                    "source": doc.metadata.get("source", ""),
                    "page": str(doc.metadata.get("page", "")),
                },
            })
            
        # Upsert the batch
        index.upsert(vectors=vectors, namespace=namespace)
        
        # Free memory before next batch
        del texts, embeddings, vectors
        gc.collect()


def pinecone_search(
    query: str, namespace: str, k: int = 10, api_key: str = None
) -> list[Document]:
    """
    Query Pinecone for the top-k most similar chunks.

    Args:
        query: Search query text.
        namespace: Document namespace to search within.
        k: Number of results.
        api_key: Pinecone API key.

    Returns:
        List of LangChain Document objects with metadata.
    """
    index = get_pinecone_index(api_key)
    embed_model = get_embeddings()

    query_embedding = embed_model.embed_query(query)

    results = index.query(
        vector=query_embedding,
        namespace=namespace,
        top_k=k,
        include_metadata=True,
    )

    docs = []
    for match in results.get("matches", []):
        meta = match.get("metadata", {})
        docs.append(Document(
            page_content=meta.get("text", ""),
            metadata={
                "chunk_index": meta.get("chunk_index", 0),
                "source": meta.get("source", ""),
                "page": meta.get("page", ""),
                "score": match.get("score", 0),
            },
        ))
    return docs


def delete_pinecone_namespace(namespace: str, api_key: str = None):
    """Delete all vectors in a namespace (cleanup on session delete)."""
    try:
        index = get_pinecone_index(api_key)
        index.delete(delete_all=True, namespace=namespace)
    except Exception:
        pass  # Best-effort cleanup
