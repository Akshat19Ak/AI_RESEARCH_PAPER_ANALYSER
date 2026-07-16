# Backend — FastAPI RAG Pipeline

The backend is a high-performance FastAPI REST API that handles document processing, hybrid retrieval, and LLM-powered analysis.

## Quick Start

```bash
# From the project root
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

pip install -r backend/requirements.txt

cp backend/.env.example .env
# Edit .env with your Groq API key

python -m uvicorn backend.api.server:app --reload --port 8000
```

- **Health Check:** http://localhost:8000/health
- **API Docs:** http://localhost:8000/docs

## Architecture

```
backend/
├── api/
│   ├── server.py          # FastAPI endpoints
│   ├── models.py          # Pydantic request/response schemas
│   └── session_manager.py # In-memory session storage
└── src/
    ├── ingestion/         # PDF/URL loading, text chunking
    ├── retrieval/         # FAISS, Pinecone, BM25, hybrid search, reranking
    ├── generation/        # LLM setup, prompts, analysis chains
    ├── evaluation/        # RAG quality metrics
    └── utils/             # Configuration, helpers
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Groq API key for LLM inference |
| `PINECONE_API_KEY` | No | Pinecone API key (optional, for cloud vector store) |
| `VECTOR_STORE_TYPE` | No | `faiss` (default) or `pinecone` |
