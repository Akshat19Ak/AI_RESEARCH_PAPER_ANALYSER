"""
llm.py — LLM (Large Language Model) setup using Groq.

WHY GROQ + LLaMA 3.3 70B:
    - 70 billion parameters → understands complex academic language
    - Groq's LPU hardware = 10-20× faster than GPU inference
    - Free tier is generous for personal/educational use
    - Open-source model (no vendor lock-in)

TEMPERATURE = 0.2:
    Lower temperature = more factual, deterministic outputs.
    For research paper analysis, we want PRECISION over creativity.
    (Creative writing would use 0.7-1.0)
"""

from langchain_groq import ChatGroq
from src.utils.config import LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS


def get_llm(api_key: str, temperature: float = None) -> ChatGroq:
    """
    Create a Groq LLM instance.

    We create a FRESH instance per interaction (not cached) because:
    - API keys might change during a session
    - Temperature might vary per analysis mode
    - LLM instances are lightweight (no model download)

    Args:
        api_key: Groq API key (from .env or user input).
        temperature: Override default temperature (optional).

    Returns:
        ChatGroq instance ready for .invoke() calls.
    """
    return ChatGroq(
        api_key=api_key,
        model=LLM_MODEL,
        temperature=temperature or LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )
