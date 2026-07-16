"""
server.py — FastAPI REST backend for the RAG pipeline.

Endpoints:
    POST /upload        Upload & index a document (PDF, Image)
    POST /upload-url    Scrape & index a URL/ArXiv page
    POST /summary       Generate structured summary
    POST /chat          Q&A with citations + evaluation metrics
    POST /insights      5 key takeaways
    POST /deepdive      Section-level analysis
    POST /interview     Generate interview questions
    POST /compare       Compare two papers
    POST /configure     Set API keys from frontend
    GET  /metrics       Evaluation metrics history
    GET  /health        Health check with config status
    DELETE /session      Clear session

Swagger docs: http://localhost:8000/docs
"""

import os
os.environ.setdefault(
    "USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from backend.api.models import (
    URLUploadRequest, ChatRequest, SummaryRequest, InsightsRequest,
    DeepDiveRequest, InterviewRequest, CompareRequest,
    UploadResponse, AnalysisResponse, ChatResponse, CompareResponse,
    MetricsResponse, HealthResponse, SourceChunk,
)
from backend.api.session_manager import sessions

from backend.src.utils.config import VECTOR_STORE_TYPE, PINECONE_API_KEY, PINECONE_NAMESPACE_PREFIX
from backend.src.utils.helpers import file_hash, clean_text
from backend.src.ingestion.pdf_loader import load_pdf
from backend.src.ingestion.url_loader import load_url
from backend.src.ingestion.chunker import chunk_text
from backend.src.retrieval.vector_store import build_vector_store
from backend.src.retrieval.bm25_store import BM25Store
from backend.src.generation.llm import get_llm
from backend.src.generation.chains import (
    run_structured_summary, run_chat, run_quick_insights,
    run_section_dive, run_interview_prep, run_comparison,
)
from backend.src.evaluation.metrics import compute_all_metrics

# ══════════════════════════════════════════════════════════════════════════════
#  APP SETUP
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="AI Research Paper Analyzer API",
    description="Industry-grade RAG pipeline: Hybrid Retrieval (BM25+FAISS/Pinecone), "
                "Cross-Encoder Reranking, Evaluation Metrics. PDF/Image/URL.",
    version="3.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════════════════════════
#  RUNTIME CONFIG (can be updated from frontend)
# ══════════════════════════════════════════════════════════════════════════════

runtime_config = {
    "groq_api_key": os.getenv("GROQ_API_KEY", ""),
    "pinecone_api_key": os.getenv("PINECONE_API_KEY", ""),
    "vector_store_type": VECTOR_STORE_TYPE,
}


class ConfigureRequest(BaseModel):
    groq_api_key: str = ""
    pinecone_api_key: str = ""
    vector_store_type: str = ""


def _get_api_key(request_key: str = "") -> str:
    """Get Groq API key: request body > runtime config > env."""
    return request_key or runtime_config["groq_api_key"]


def _get_pinecone_key() -> str:
    """Get Pinecone API key: runtime config > env."""
    return runtime_config["pinecone_api_key"]


def _use_pinecone() -> bool:
    """Check if Pinecone should be used."""
    return runtime_config["vector_store_type"] == "pinecone" and bool(_get_pinecone_key())


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _docs_to_sources(docs) -> list[SourceChunk]:
    sources = []
    for doc in docs:
        sources.append(SourceChunk(
            chunk_index=doc.metadata.get("chunk_index", 0),
            content=doc.page_content[:500],
            source=doc.metadata.get("source", ""),
            page=str(doc.metadata.get("page", "")) or None,
            reranker_score=doc.metadata.get("reranker_score"),
        ))
    return sources


def _build_indices(chunks, session_id: str):
    """Build vector + BM25 indices. Uses Pinecone if configured, else FAISS."""
    bm25 = BM25Store(chunks)

    if _use_pinecone():
        from backend.src.retrieval.pinecone_store import upsert_to_pinecone
        namespace = f"{PINECONE_NAMESPACE_PREFIX}{session_id}"
        upsert_to_pinecone(chunks, namespace, api_key=_get_pinecone_key())
        vectorstore = None  # Pinecone handles dense search
    else:
        vectorstore = build_vector_store(chunks)

    return vectorstore, bm25


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT: Health Check
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    return HealthResponse(
        status="ok",
        active_sessions=sessions.active_count,
        version="3.0",
        groq_configured=bool(runtime_config["groq_api_key"]),
        pinecone_configured=bool(runtime_config["pinecone_api_key"]),
        vector_store_type=runtime_config["vector_store_type"],
    )


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT: Configure API Keys (from frontend)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/configure", tags=["System"])
async def configure_keys(req: ConfigureRequest):
    """Set API keys from the frontend UI (not persisted to disk)."""
    if req.groq_api_key:
        runtime_config["groq_api_key"] = req.groq_api_key
    if req.pinecone_api_key:
        runtime_config["pinecone_api_key"] = req.pinecone_api_key
    if req.vector_store_type in ("pinecone", "faiss"):
        runtime_config["vector_store_type"] = req.vector_store_type
    return {
        "status": "configured",
        "groq_configured": bool(runtime_config["groq_api_key"]),
        "pinecone_configured": bool(runtime_config["pinecone_api_key"]),
        "vector_store_type": runtime_config["vector_store_type"],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT: Upload Document (File)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/upload", response_model=UploadResponse, tags=["Document"])
async def upload_document(
    file: UploadFile = File(...),
    api_key: str = Form(""),
):
    try:
        file_bytes = await file.read()
        filename = file.filename or "upload"
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

        if ext == "pdf":
            raw_text, meta = load_pdf(file_bytes, filename)
        else:
            raise HTTPException(400, f"Unsupported: .{ext}. Use PDF.")

        raw_text = clean_text(raw_text)
        meta["char_count"] = len(raw_text)
        if len(raw_text.strip()) < 50:
            raise HTTPException(400, "Could not extract enough text from the document.")

        chunks = chunk_text(raw_text, meta)
        session_id = sessions.create_session()
        vectorstore, bm25 = _build_indices(chunks, session_id)

        session = sessions.get(session_id)
        session.vector_store = vectorstore
        session.bm25_store = bm25
        session.doc_metadata = meta
        session.doc_chunks = chunks
        session.raw_text = raw_text
        session.doc_hash = file_hash(file_bytes)
        session.use_pinecone = _use_pinecone()

        return UploadResponse(
            session_id=session_id, status="success", metadata=meta,
            chunk_count=len(chunks),
            message=f"Indexed {len(chunks)} chunks from '{filename}'"
                    + (" [Pinecone]" if _use_pinecone() else " [FAISS]"),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Processing failed: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT: Upload URL
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/upload-url", response_model=UploadResponse, tags=["Document"])
async def upload_url_endpoint(req: URLUploadRequest):
    try:
        raw_text, meta = load_url(req.url)
        raw_text = clean_text(raw_text)
        meta["char_count"] = len(raw_text)
        if len(raw_text.strip()) < 50:
            raise HTTPException(400, "Could not extract enough text from URL.")

        chunks = chunk_text(raw_text, meta)
        session_id = sessions.create_session()
        vectorstore, bm25 = _build_indices(chunks, session_id)

        session = sessions.get(session_id)
        session.vector_store = vectorstore
        session.bm25_store = bm25
        session.doc_metadata = meta
        session.doc_chunks = chunks
        session.raw_text = raw_text
        session.doc_hash = file_hash(req.url.encode())
        session.use_pinecone = _use_pinecone()

        return UploadResponse(
            session_id=session_id, status="success", metadata=meta,
            chunk_count=len(chunks),
            message=f"Indexed {len(chunks)} chunks from URL"
                    + (" [Pinecone]" if _use_pinecone() else " [FAISS]"),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"URL processing failed: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
#  RETRIEVAL HELPER (Pinecone or FAISS based on session)
# ══════════════════════════════════════════════════════════════════════════════

def _get_retrieval_stores(session):
    """Return (vectorstore_or_None, bm25, pinecone_namespace_or_None)."""
    if getattr(session, "use_pinecone", False):
        namespace = f"{PINECONE_NAMESPACE_PREFIX}{session.session_id}"
        return None, session.bm25_store, namespace
    return session.vector_store, session.bm25_store, None


def _run_with_retrieval(session, llm, chain_fn, *args):
    """Execute a chain with Pinecone or FAISS retrieval."""
    vs, bm25, ns = _get_retrieval_stores(session)

    if ns:
        # Pinecone mode: patch dense_search to use Pinecone
        from backend.src.retrieval import pinecone_store
        import backend.src.retrieval.vector_store as vs_mod
        original_dense = vs_mod.dense_search

        def pinecone_dense(vectorstore, query, k=10):
            return pinecone_store.pinecone_search(
                query, ns, k=k, api_key=_get_pinecone_key()
            )
        vs_mod.dense_search = pinecone_dense
        try:
            result = chain_fn(llm, vs, bm25, *args)
        finally:
            vs_mod.dense_search = original_dense
        return result
    else:
        return chain_fn(llm, vs, bm25, *args)


# ══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/summary", response_model=AnalysisResponse, tags=["Analysis"])
async def generate_summary(req: SummaryRequest):
    try:
        session = sessions.get(req.session_id)
        api_key = _get_api_key(req.api_key)
        if not api_key:
            raise HTTPException(400, "Groq API key not configured.")
        llm = get_llm(api_key)
        result, docs = _run_with_retrieval(session, llm, run_structured_summary)
        session.structured_summary = result
        return AnalysisResponse(result=result, sources=_docs_to_sources(docs))
    except KeyError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Summary failed: {str(e)}")


@app.post("/chat", response_model=ChatResponse, tags=["Analysis"])
async def chat_qa(req: ChatRequest):
    try:
        session = sessions.get(req.session_id)
        api_key = _get_api_key(req.api_key)
        if not api_key:
            raise HTTPException(400, "Groq API key not configured.")
        llm = get_llm(api_key)

        result = _run_with_retrieval(
            session, llm, run_chat,
            req.question, session.chat_history, req.difficulty,
        )
        answer, docs, updated_history = result
        session.chat_history = updated_history

        metrics = compute_all_metrics(req.question, answer, docs)
        session.metrics_history.append(metrics)

        return ChatResponse(
            answer=answer, sources=_docs_to_sources(docs),
            metrics=metrics, history_length=len(session.chat_history),
        )
    except KeyError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Chat failed: {str(e)}")


@app.post("/insights", response_model=AnalysisResponse, tags=["Analysis"])
async def quick_insights(req: InsightsRequest):
    try:
        session = sessions.get(req.session_id)
        api_key = _get_api_key(req.api_key)
        if not api_key:
            raise HTTPException(400, "Groq API key not configured.")
        llm = get_llm(api_key)
        result, docs = _run_with_retrieval(session, llm, run_quick_insights)
        return AnalysisResponse(result=result, sources=_docs_to_sources(docs))
    except KeyError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Insights failed: {str(e)}")


@app.post("/deepdive", response_model=AnalysisResponse, tags=["Analysis"])
async def section_deepdive(req: DeepDiveRequest):
    try:
        session = sessions.get(req.session_id)
        api_key = _get_api_key(req.api_key)
        if not api_key:
            raise HTTPException(400, "Groq API key not configured.")
        llm = get_llm(api_key)
        result, docs = _run_with_retrieval(session, llm, run_section_dive)
        return AnalysisResponse(result=result, sources=_docs_to_sources(docs))
    except KeyError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Deep-dive failed: {str(e)}")


@app.post("/interview", response_model=AnalysisResponse, tags=["Analysis"])
async def interview_prep(req: InterviewRequest):
    try:
        session = sessions.get(req.session_id)
        api_key = _get_api_key(req.api_key)
        if not api_key:
            raise HTTPException(400, "Groq API key not configured.")
        llm = get_llm(api_key)
        result, docs = _run_with_retrieval(session, llm, run_interview_prep)
        return AnalysisResponse(result=result, sources=_docs_to_sources(docs))
    except KeyError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Interview prep failed: {str(e)}")


@app.post("/compare", response_model=CompareResponse, tags=["Analysis"])
async def compare_papers(req: CompareRequest):
    try:
        session = sessions.get(req.session_id)
        api_key = _get_api_key(req.api_key)
        if not api_key:
            raise HTTPException(400, "Groq API key not configured.")
        llm = get_llm(api_key)
        vs, bm25, ns = _get_retrieval_stores(session)
        if ns:
            from backend.src.retrieval import pinecone_store
            import backend.src.retrieval.vector_store as vs_mod
            original = vs_mod.dense_search
            def pc_dense(vectorstore, query, k=10):
                return pinecone_store.pinecone_search(query, ns, k=k, api_key=_get_pinecone_key())
            vs_mod.dense_search = pc_dense
            try:
                result = run_comparison(llm, vs, bm25, req.paper1_summary)
            finally:
                vs_mod.dense_search = original
        else:
            result = run_comparison(llm, vs, bm25, req.paper1_summary)
        return CompareResponse(result=result)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Comparison failed: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT: Flowchart Generation (Mermaid)
# ══════════════════════════════════════════════════════════════════════════════

FLOWCHART_PROMPT = """Based on the following research paper content, generate a Mermaid flowchart diagram
that shows the paper's methodology/pipeline/architecture.

Use valid Mermaid syntax with the `graph TD` format. Include:
- Main stages of the methodology
- Key components and their relationships
- Data flow between components

CRITICAL RULES FOR MERMAID SYNTAX:
1. NEVER use parentheses `()` or square brackets `[]` inside node names/labels.
   BAD:  A[Encoder (Self-Attention)]
   GOOD: A[Encoder Self-Attention]
2. Keep it clean and readable with 8-15 nodes maximum.
3. Return ONLY the Mermaid code, starting with `graph TD`. No markdown fences, no explanation.

Content:
{context}
"""


class FlowchartRequest(BaseModel):
    session_id: str
    api_key: str = ""


@app.post("/flowchart", tags=["Analysis"])
async def generate_flowchart(req: FlowchartRequest):
    """Generate a Mermaid flowchart of the paper's methodology."""
    try:
        session = sessions.get(req.session_id)
        api_key = _get_api_key(req.api_key)
        if not api_key:
            raise HTTPException(400, "Groq API key not configured.")
        llm = get_llm(api_key)

        # Get relevant methodology chunks
        vs, bm25, ns = _get_retrieval_stores(session)
        query = "methodology architecture pipeline approach system design workflow"

        if ns:
            from backend.src.retrieval import pinecone_store
            docs = pinecone_store.pinecone_search(query, ns, k=8, api_key=_get_pinecone_key())
        else:
            from backend.src.retrieval.hybrid_retriever import hybrid_search
            from backend.src.retrieval.reranker import rerank
            docs = hybrid_search(vs, bm25, query, k=10)
            docs = rerank(query, docs, top_k=6)

        context = "\n\n".join([d.page_content for d in docs])
        prompt = FLOWCHART_PROMPT.format(context=context[:4000])
        result = llm.invoke(prompt)
        mermaid_code = result.content.strip()

        # Clean up: remove markdown fences if LLM added them
        import re
        match = re.search(r"```(?:mermaid)?\s*(.*?)\s*```", mermaid_code, re.DOTALL | re.IGNORECASE)
        if match:
            mermaid_code = match.group(1).strip()
        
        # Sometimes LLMs prepend 'mermaid\n' without fences
        if mermaid_code.lower().startswith("mermaid\n"):
            mermaid_code = mermaid_code[8:].strip()

        return {"mermaid": mermaid_code, "sources": [s.dict() for s in _docs_to_sources(docs)]}
    except KeyError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Flowchart generation failed: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
#  METRICS + HISTORY + SESSION MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/metrics/{session_id}", response_model=MetricsResponse, tags=["Evaluation"])
async def get_metrics(session_id: str):
    try:
        session = sessions.get(session_id)
        ml = session.metrics_history
        averages = {}
        if ml:
            for key in ["answer_relevance", "faithfulness", "context_precision", "overall_score"]:
                vals = [m[key] for m in ml]
                averages[key] = round(sum(vals) / len(vals), 1)
        return MetricsResponse(session_metrics=ml, averages=averages)
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.get("/history/{session_id}", tags=["Analysis"])
async def get_chat_history(session_id: str):
    try:
        session = sessions.get(session_id)
        return {"history": session.chat_history}
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.delete("/session/{session_id}", tags=["System"])
async def delete_session(session_id: str):
    try:
        session = sessions.get(session_id)
        if getattr(session, "use_pinecone", False):
            from backend.src.retrieval.pinecone_store import delete_pinecone_namespace
            delete_pinecone_namespace(
                f"{PINECONE_NAMESPACE_PREFIX}{session_id}",
                api_key=_get_pinecone_key(),
            )
    except KeyError:
        pass
    deleted = sessions.delete(session_id)
    if deleted:
        return {"status": "deleted", "session_id": session_id}
    raise HTTPException(404, f"Session '{session_id}' not found.")
