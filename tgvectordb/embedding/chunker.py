"""
splits documents into chunks suitable for embedding.

nothing fancy here - just paragraph-aware splitting with overlap.
tried to keep it simple because chunking strategies are a whole
rabbit hole and the user can always pre-chunk if they want.
"""

import re
from pathlib import Path
from typing import Optional


def chunk_text(
    text: str,
    chunk_size: int = 400,  # roughly in words
    overlap: int = 50,
    source: str = "",
) -> list:
    """
    split text into overlapping chunks.

    tries to break on paragraph boundaries first,
    falls back to sentence boundaries, then just word count.

    returns list of dicts: [{"text": "...", "src": "...", "chunk_idx": 0}, ...]
    """
    if not text or not text.strip():
        return []

    # normalize whitespace but keep paragraph breaks
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    # split into paragraphs first
    paragraphs = re.split(r"\n\n+", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current_chunk_words = []
    current_word_count = 0

    for para in paragraphs:
        para_words = para.split()
        para_len = len(para_words)

        # if this single paragraph is bigger than chunk_size, split it further
        if para_len > chunk_size:
            # flush what we have
            if current_chunk_words:
                chunks.append(" ".join(current_chunk_words))
                # keep overlap
                overlap_words = current_chunk_words[-overlap:] if overlap > 0 else []
                current_chunk_words = list(overlap_words)
                current_word_count = len(current_chunk_words)

            # split the big paragraph by sentences
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                sent_words = sent.split()
                if current_word_count + len(sent_words) > chunk_size and current_chunk_words:
                    chunks.append(" ".join(current_chunk_words))
                    overlap_words = current_chunk_words[-overlap:] if overlap > 0 else []
                    current_chunk_words = list(overlap_words)
                    current_word_count = len(current_chunk_words)

                current_chunk_words.extend(sent_words)
                current_word_count += len(sent_words)
        else:
            # normal case - add paragraph to current chunk
            if current_word_count + para_len > chunk_size and current_chunk_words:
                chunks.append(" ".join(current_chunk_words))
                overlap_words = current_chunk_words[-overlap:] if overlap > 0 else []
                current_chunk_words = list(overlap_words)
                current_word_count = len(current_chunk_words)

            current_chunk_words.extend(para_words)
            current_word_count += para_len

    # dont forget the last chunk
    if current_chunk_words:
        chunks.append(" ".join(current_chunk_words))

    # build the result with metadata
    result = []
    for idx, chunk_text in enumerate(chunks):
        if not chunk_text.strip():
            continue
        result.append({
            "text": chunk_text.strip(),
            "src": source,
            "chunk_idx": idx,
        })

    return result


def read_file_as_text(filepath: str) -> str:
    """
    read a file and return its text content.
    supports .txt, .md, and .pdf (if pdfplumber is installed).
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"cant find file: {filepath}")

    ext = path.suffix.lower()

    if ext in (".txt", ".md", ".text", ".rst"):
        return path.read_text(encoding="utf-8", errors="replace")

    elif ext == ".pdf":
        return _extract_pdf_text(str(path))

    else:
        # try reading as plain text, might work for csv, json, etc
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            raise ValueError(
                f"dont know how to read {ext} files. "
                f"supported: .txt, .md, .pdf"
            )


def _extract_pdf_text(pdf_path: str) -> str:
    """pull text out of a pdf. needs pdfplumber installed."""
    try:
        import pdfplumber
    except ImportError:
        raise ImportError(
            "pdfplumber is needed for PDF support. install with:\n"
            "  pip install pdfplumber\n"
            "  or: pip install tgvectordb[pdf]"
        )

    all_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                all_text.append(page_text)

    return "\n\n".join(all_text)


def chunk_file(filepath: str, chunk_size: int = 400, overlap: int = 50) -> list:
    """convenience function - read a file and chunk it in one go."""
    text = read_file_as_text(filepath)
    source = Path(filepath).name
    return chunk_text(text, chunk_size=chunk_size, overlap=overlap, source=source)
