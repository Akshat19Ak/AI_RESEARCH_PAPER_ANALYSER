"""
config.py — Centralized configuration for the RAG pipeline.

All tunable parameters live here so you never hunt through code to change settings.
Each constant has a comment explaining WHY that value was chosen.
"""

import os
from dotenv import load_dotenv

# ─── Load .env file (contains GROQ_API_KEY) ─────────────────────────────────
load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
#  LLM CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 3000

# ══════════════════════════════════════════════════════════════════════════════
#  PINECONE (CLOUD VECTOR STORE)
# ══════════════════════════════════════════════════════════════════════════════

# "pinecone" = cloud-hosted, scalable | "faiss" = local, in-memory
VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "faiss")

# Pinecone free tier: 1 serverless index, 100K vectors
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-analyzer")
PINECONE_NAMESPACE_PREFIX = "session-"

# ══════════════════════════════════════════════════════════════════════════════
#  EMBEDDING MODEL
# ══════════════════════════════════════════════════════════════════════════════

# MiniLM-L6-v2: Microsoft's compressed BERT (only 23MB).
# Runs locally on CPU — zero API cost, works offline.
# Produces 384-dimensional vectors for semantic similarity.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"

# ══════════════════════════════════════════════════════════════════════════════
#  RERANKER MODEL (Cross-Encoder)
# ══════════════════════════════════════════════════════════════════════════════

# Cross-encoder reranker: scores query-document PAIRS (not independent embeddings).
# This gives much more accurate relevance scores than bi-encoder similarity.
# Only ~23MB — very lightweight for the quality boost it provides.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ══════════════════════════════════════════════════════════════════════════════
#  TEXT CHUNKING PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

# 1000 chars ≈ 250 tokens. This size:
#   - Preserves enough context within each chunk for meaningful retrieval
#   - Is small enough to fit 8-12 chunks in LLM context window
#   - Works well with MiniLM-L6-v2 (max 512 tokens input)
CHUNK_SIZE = 1000

# 200-char overlap prevents information loss at chunk boundaries.
# Example: A sentence split across two chunks will appear in BOTH.
CHUNK_OVERLAP = 200

# Splitting priority: try paragraph breaks first, then sentences, then words.
# This preserves semantic coherence within each chunk.
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

# ══════════════════════════════════════════════════════════════════════════════
#  RETRIEVAL PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

# How many chunks to retrieve from EACH retriever (BM25 and FAISS).
# More candidates = better chance of finding the best chunks.
RETRIEVAL_TOP_K = 10

# How many chunks to keep AFTER reranking for the final LLM prompt.
# 4-6 chunks balances context richness vs token cost.
FINAL_TOP_K = 5

# Weight for combining BM25 (sparse) and FAISS (dense) scores.
# 0.5 = equal weight. Increase for more keyword-matching, decrease for more semantic.
HYBRID_ALPHA = 0.5

# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION METRIC THRESHOLDS
# ══════════════════════════════════════════════════════════════════════════════

# A sentence is considered "faithful" (grounded in source) if its embedding
# similarity to the best-matching source chunk exceeds this threshold.
FAITHFULNESS_THRESHOLD = 0.45

# User-Agent string to avoid being blocked when scraping URLs.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
)

# Set USER_AGENT environment variable before any LangChain imports
os.environ.setdefault("USER_AGENT", USER_AGENT)
