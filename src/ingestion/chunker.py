"""
chunker.py — Text chunking with metadata injection.

WHY WE CHUNK:
    1. LLMs have token limits — we can't send 50,000 chars at once
    2. Smaller chunks = more precise similarity search results
    3. Only RELEVANT chunks go to the LLM, not the whole document

HOW IT WORKS:
    RecursiveCharacterTextSplitter tries to split at natural boundaries:
    Paragraphs (\n\n) → Lines (\n) → Sentences (. ) → Words ( ) → Characters

    This preserves semantic coherence: a chunk about "self-attention"
    won't be randomly cut mid-sentence.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.utils.config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_SEPARATORS


def chunk_text(text: str, metadata: dict) -> list[Document]:
    """
    Split document text into overlapping chunks with rich metadata.

    Each chunk gets metadata injected for:
        - Citation tracking (chunk_index, source)
        - Debugging (chunk_preview)
        - Statistics (total chunk_count)

    Args:
        text: Full document text to split.
        metadata: Document-level metadata (source, type, pages, etc.).

    Returns:
        List of LangChain Document objects, each containing:
            - page_content: The chunk text
            - metadata: Merged document + chunk metadata
    """
    # ── Create the splitter with our configured parameters ───────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,          # 1000 chars per chunk
        chunk_overlap=CHUNK_OVERLAP,    # 200 char overlap between adjacent chunks
        separators=CHUNK_SEPARATORS,    # Split at paragraphs first, then sentences
        length_function=len,
    )

    # ── Split text into raw strings ─────────────────────────────────────
    raw_chunks = splitter.split_text(text)

    # ── Wrap each chunk in a Document with rich metadata ────────────────
    # This metadata is crucial for the CITATION system:
    # When the LLM references "[Chunk 5]", we can look up exactly
    # which part of the document that corresponds to.
    docs = []
    for i, chunk in enumerate(raw_chunks):
        chunk_meta = {
            **metadata,                             # Inherit document-level metadata
            "chunk_index": i,                       # Position in document
            "chunk_count": len(raw_chunks),          # Total chunks
            "chunk_preview": chunk[:80].replace("\n", " "),  # Quick preview
        }
        docs.append(Document(page_content=chunk, metadata=chunk_meta))

    return docs
