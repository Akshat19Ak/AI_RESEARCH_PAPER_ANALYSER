"""
models.py — Pydantic request/response schemas for the API.

Every endpoint has typed input/output models. This gives us:
    1. Automatic validation (wrong types → clear error messages)
    2. Auto-generated Swagger/OpenAPI docs at /docs
    3. Type safety across the backend ↔ frontend boundary
"""

from pydantic import BaseModel, Field
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
#  REQUEST MODELS (what the frontend sends)
# ══════════════════════════════════════════════════════════════════════════════

class URLUploadRequest(BaseModel):
    """Request to process a URL/ArXiv document."""
    url: str = Field(..., description="Full URL to scrape", examples=["https://arxiv.org/html/2401.00001"])
    api_key: str = Field(..., description="Groq API key")


class ChatRequest(BaseModel):
    """Request for Chat Q&A."""
    session_id: str
    question: str
    difficulty: str = Field(default="Expert", description="Beginner or Expert")
    api_key: str


class SummaryRequest(BaseModel):
    """Request to generate a structured summary."""
    session_id: str
    api_key: str


class InsightsRequest(BaseModel):
    """Request to extract quick insights."""
    session_id: str
    api_key: str


class DeepDiveRequest(BaseModel):
    """Request for full paper deep-dive."""
    session_id: str
    api_key: str


class InterviewRequest(BaseModel):
    """Request to generate interview questions."""
    session_id: str
    api_key: str


class CompareRequest(BaseModel):
    """Request to compare current paper with a saved summary."""
    session_id: str
    paper1_summary: str = Field(..., description="Structured summary of Paper 1")
    api_key: str


# ══════════════════════════════════════════════════════════════════════════════
#  RESPONSE MODELS (what the API returns)
# ══════════════════════════════════════════════════════════════════════════════

class SourceChunk(BaseModel):
    """A single retrieved source chunk with metadata."""
    chunk_index: int = 0
    content: str
    source: str = ""
    page: Optional[str] = None
    reranker_score: Optional[float] = None


class UploadResponse(BaseModel):
    """Response after successful document upload + indexing."""
    session_id: str
    status: str = "success"
    metadata: dict = {}
    chunk_count: int = 0
    message: str = ""


class AnalysisResponse(BaseModel):
    """Response for summary, insights, deep-dive, interview endpoints."""
    result: str
    sources: list[SourceChunk] = []


class ChatResponse(BaseModel):
    """Response for chat Q&A with evaluation metrics."""
    answer: str
    sources: list[SourceChunk] = []
    metrics: dict = {}
    history_length: int = 0


class CompareResponse(BaseModel):
    """Response for paper comparison."""
    result: str


class MetricsResponse(BaseModel):
    """Response for evaluation metrics history."""
    session_metrics: list[dict] = []
    averages: dict = {}


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    active_sessions: int = 0
    version: str = "3.0"
    groq_configured: bool = False
    pinecone_configured: bool = False
    vector_store_type: str = "faiss"
