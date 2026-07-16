"""
pdf_loader.py — PDF text extraction with 2-pass fallback strategy.

Strategy:
    Pass 1: pypdf        → Fast, handles most modern text-layer PDFs
    Pass 2: pdfplumber   → Better layout extraction for dense tables/columns

This graceful degradation ensures maximum text extraction from any standard PDF.
"""

import io


def load_pdf(file_bytes: bytes, filename: str) -> tuple[str, dict]:
    """
    Extract text from a PDF file using a 2-pass fallback strategy.

    Args:
        file_bytes: Raw bytes of the uploaded PDF file.
        filename: Original filename (used in metadata for citations).

    Returns:
        Tuple of (extracted_text, metadata_dict).

    Raises:
        ValueError: If both passes fail to extract readable text.
    """
    text = ""
    meta = {"source": filename, "type": "pdf", "pages": 0}

    # ── PASS 1: pypdf (fastest, works for most modern PDFs) ──────────────
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(file_bytes))
        meta["pages"] = len(reader.pages)

        page_texts = []
        for page_num, page in enumerate(reader.pages, 1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                page_texts.append(page_text)

        text = "\n".join(page_texts).strip()
    except Exception:
        pass

    # ── PASS 2: pdfplumber (better for dense layouts, tables, columns) ───
    if len(text) < 500:
        try:
            import pdfplumber

            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                meta["pages"] = len(pdf.pages)
                text = "\n".join(
                    p.extract_text() or "" for p in pdf.pages
                ).strip()
        except ImportError:
            pass  # pdfplumber not installed — skip
        except Exception:
            pass

    # ── Validation ───────────────────────────────────────────────────────
    if not text.strip():
        raise ValueError(
            "❌ Could not extract text from this PDF.\n"
            "• Text-layer extraction failed\n"
            "• pdfplumber extraction failed\n\n"
            "Try: Ensure the PDF is not password-protected and is not a scanned image."
        )

    meta["char_count"] = len(text)
    return text, meta



