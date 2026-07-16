"""
session_manager.py — In-memory session storage for document indices.

Each uploaded document gets a unique session ID. The session stores:
    - FAISS vector store (dense index)
    - BM25 store (sparse index)
    - Document metadata (pages, char_count, type, etc.)
    - Document chunks (for evaluation metrics)
    - Chat history (for multi-turn conversation)
    - Evaluation metrics history

WHY IN-MEMORY (not a database):
    - This is a single-user research tool, not a multi-tenant SaaS
    - FAISS indices are numpy arrays — not easily serializable to SQL
    - In-memory = instant access, zero latency
    - For production: use Redis or a proper vector DB (Pinecone, Weaviate)
"""

import uuid
from dataclasses import dataclass, field
from langchain_community.vectorstores import FAISS
from backend.src.retrieval.bm25_store import BM25Store


@dataclass
class Session:
    """Holds all state for one document analysis session."""
    session_id: str
    vector_store: FAISS = None
    bm25_store: BM25Store = None
    doc_metadata: dict = field(default_factory=dict)
    doc_chunks: list = field(default_factory=list)
    raw_text: str = ""
    doc_hash: str = ""
    chat_history: list = field(default_factory=list)
    metrics_history: list = field(default_factory=list)
    structured_summary: str = ""
    use_pinecone: bool = False


class SessionManager:
    """
    Manages active document sessions.

    Thread-safe for FastAPI's async handlers because each session
    is independent — no shared mutable state between sessions.
    """

    def __init__(self):
        self._sessions: dict[str, Session] = {}

    def create_session(self) -> str:
        """Create a new empty session and return its ID."""
        session_id = str(uuid.uuid4())[:8]  # Short ID for convenience
        self._sessions[session_id] = Session(session_id=session_id)
        return session_id

    def get(self, session_id: str) -> Session:
        """
        Retrieve a session by ID.

        Raises:
            KeyError: If session_id doesn't exist.
        """
        if session_id not in self._sessions:
            raise KeyError(f"Session '{session_id}' not found. Upload a document first.")
        return self._sessions[session_id]

    def delete(self, session_id: str) -> bool:
        """Delete a session and free its memory."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    @property
    def active_count(self) -> int:
        """Number of active sessions."""
        return len(self._sessions)


# Global singleton — shared across all FastAPI endpoints
sessions = SessionManager()
