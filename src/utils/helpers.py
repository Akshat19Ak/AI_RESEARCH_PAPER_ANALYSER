"""
helpers.py — Reusable utility functions.

These are small, stateless functions used across the pipeline:
  - File hashing (detect duplicate uploads)
  - Context formatting (prepare chunks for LLM prompt)
  - Chat history formatting
"""

import hashlib
from langchain_core.documents import Document


# ══════════════════════════════════════════════════════════════════════════════
#  FILE HASHING
# ══════════════════════════════════════════════════════════════════════════════

def file_hash(content: bytes) -> str:
    """
    Compute SHA-256 hash of file content.

    WHY: If the user uploads the same document twice, we skip re-processing
    by comparing hashes. This saves ~5-10 seconds per duplicate upload.

    Args:
        content: Raw bytes of the uploaded file (or URL string encoded to bytes).

    Returns:
        64-character hex string (SHA-256 digest).
    """
    return hashlib.sha256(content).hexdigest()


# ══════════════════════════════════════════════════════════════════════════════
#  CONTEXT FORMATTING
# ══════════════════════════════════════════════════════════════════════════════

def format_context(docs: list[Document]) -> str:
    """
    Format retrieved Document chunks into a clean context string for the LLM.

    Each chunk is labeled with its source and chunk index so the LLM can
    cite specific sources in its answer (e.g., "[Chunk 3]").

    Args:
        docs: List of LangChain Document objects from retrieval.

    Returns:
        Formatted string with chunk labels and separators.
    """
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        chunk_idx = doc.metadata.get("chunk_index", "?")
        page = doc.metadata.get("page", "")
        page_str = f" | Page: {page}" if page else ""

        header = f"[Chunk {chunk_idx}{page_str} | Source: {source}]"
        parts.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(parts)


def format_history(history: list[dict], max_turns: int = 4) -> str:
    """
    Format chat history for inclusion in the LLM prompt.

    WHY only last 4 turns: Including too much history eats into the LLM's
    context window, leaving less room for retrieved document chunks.
    4 turns (8 messages) is enough for conversational continuity.

    Args:
        history: List of {"role": "user"/"assistant", "content": "..."} dicts.
        max_turns: Maximum number of conversation turns to include.

    Returns:
        Formatted conversation string.
    """
    if not history:
        return "No previous conversation."

    # Each "turn" = 1 user message + 1 assistant response
    recent = history[-(max_turns * 2):]
    lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        # Truncate long messages to save tokens
        content = msg["content"][:400]
        lines.append(f"{role}: {content}")

    return "\n".join(lines)


def clean_text(text: str) -> str:
    """
    Robust text cleanup for extracted document content.
    - Removes null bytes and non-printable control characters.
    - Normalizes all whitespace (tabs, non-breaking spaces) to regular spaces.
    - Collapses multiple newlines and spaces to prevent count inflation.
    """
    import re
    if not text:
        return ""
    
    # 1. Remove null bytes and common control characters (except newline/tab)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    
    # 2. Normalize whitespace (tabs, non-breaking spaces, etc. -> space)
    text = text.replace("\t", " ")
    text = text.replace("\xa0", " ") # Non-breaking space
    
    # 3. Collapse multiple spaces
    text = re.sub(r" +", " ", text)
    
    # 4. Clean up whitespace around newlines
    text = re.sub(r" \n", "\n", text)
    text = re.sub(r"\n ", "\n", text)
    
    # 5. Collapse excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    return text.strip()
