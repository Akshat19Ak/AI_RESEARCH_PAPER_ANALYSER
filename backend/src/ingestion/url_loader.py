"""
url_loader.py — URL content extraction with 2-pass scraping strategy.

Strategy:
    Pass 1: trafilatura  → Best for articles/papers (strips nav, ads, footers)
    Pass 2: WebBaseLoader → LangChain's general HTML scraper (broad fallback)

Includes detection for JavaScript-protected sites (Cloudflare, etc.)
that would return useless content.
"""


def load_url(url: str) -> tuple[str, dict]:
    """
    Extract readable text content from a web URL.

    Handles:
        - ArXiv HTML pages (arxiv.org/html/...)
        - Wikipedia articles
        - Medium posts, blog articles
        - Documentation sites

    Blocks:
        - Cloudflare-protected sites
        - JavaScript-required sites
        - Login-required pages

    Args:
        url: Full URL string to scrape.

    Returns:
        Tuple of (extracted_text, metadata_dict).

    Raises:
        ValueError: If the URL is blocked or returns too little content.
    """
    # ── PDF DETECTION ────────────────────────────────────────────────────
    # ── PDF DETECTION ────────────────────────────────────────────────────
    # If the user provides a PDF URL, fetch it and use the PDF loader directly.
    # This prevents scraping binary junk as HTML.
    if url.lower().split("?")[0].endswith(".pdf") or "arxiv.org/pdf/" in url.lower():
        import requests
        from backend.src.ingestion.pdf_loader import load_pdf
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            text, meta = load_pdf(response.content, url)
            meta["type"] = "url_pdf"
            return text, meta
        except Exception as e:
            raise ValueError(f"Failed to fetch or parse PDF from URL: {e}")

    meta = {"source": url, "type": "url"}
    text = ""

    # ── PASS 1: trafilatura (best for articles and papers) ───────────────
    # trafilatura is designed specifically for extracting MAIN CONTENT
    # from web pages, ignoring navigation, ads, and sidebars.
    try:
        import trafilatura

        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(
                downloaded,
                include_tables=True,
                include_comments=False,
            ) or ""
    except ImportError:
        pass  # trafilatura not installed — try fallback

    # ── PASS 2: LangChain WebBaseLoader (general fallback) ──────────────
    # Uses BeautifulSoup under the hood. Less precise than trafilatura
    # but handles a wider range of page structures.
    if len(text) < 300:
        try:
            from langchain_community.document_loaders import WebBaseLoader

            loader = WebBaseLoader(url)
            docs = loader.load()
            text = "\n\n".join(d.page_content for d in docs)
        except Exception as e:
            if not text:
                raise ValueError(f"Could not fetch URL content: {e}")

    # ── BLOCK DETECTION: Catch JavaScript-protected pages ────────────────
    # These sites return a "checking your browser" page instead of content.
    blocking_indicators = [
        "just a moment", "enable javascript", "checking your browser",
        "cloudflare", "access denied", "please enable javascript",
    ]
    text_lower = text.lower()

    if any(phrase in text_lower for phrase in blocking_indicators) or len(text) < 200:
        raise ValueError(
            "⚠️ This website blocks automated scraping.\n\n"
            "✅ Try instead:\n"
            "• ArXiv HTML: https://arxiv.org/html/...\n"
            "• Wikipedia: https://en.wikipedia.org/wiki/...\n"
            "• Semantic Scholar: https://www.semanticscholar.org/\n"
            "• Documentation sites, Medium articles, open blogs"
        )

    meta["char_count"] = len(text)
    return text, meta
