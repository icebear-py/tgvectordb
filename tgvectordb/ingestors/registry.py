"""
the main entry point for ingesting any file.

you throw a filepath at ingest() and it figures out what
kind of file it is, picks the right ingestor, extracts the text,
chunks it, and gives you back a nice list of chunks ready for embedding.

supported formats:
  - .pdf (needs pdfplumber)
  - .docx (needs python-docx)
  - .txt, .md, .rst, .log (plain text)
  - .html, .htm (strips tags)
  - .csv, .tsv (turns rows into readable text)
  - .json, .jsonl (extracts text fields)
  - .py, .js, .java, .go, etc (code files)
  - .xml, .yaml, .toml, .ini (config files)

basically if its text-based, we can probably read it.
"""

from pathlib import Path
from typing import Optional

from tgvectordb.ingestors import text_ingestor, pdf_ingestor, docx_ingestor
from tgvectordb.embedding.chunker import chunk_text


# order matters - we check more specific ones first
_INGESTORS = [
    pdf_ingestor,
    docx_ingestor,
    text_ingestor,  # text is the fallback, handles most things
]


def ingest(
    filepath: str,
    chunk_size: int = 400,
    overlap: int = 50,
) -> list:
    """
    read a file, extract its text, and chunk it for embedding.

    this is the main function you'd use. give it a path,
    get back a list of chunks with metadata.

    args:
        filepath: path to the file
        chunk_size: approximate words per chunk
        overlap: words of overlap between chunks

    returns:
        list of dicts like:
        [{"text": "...", "src": "paper.pdf", "chunk_idx": 0, "format": "pdf"}, ...]
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"cant find: {filepath}")

    ext = path.suffix.lower()

    # find the right ingestor
    ingestor = _find_ingestor(filepath)
    if ingestor is None:
        raise ValueError(
            f"dont know how to read '{ext}' files.\n"
            f"supported: .pdf, .docx, .txt, .md, .html, .csv, .json, "
            f"and most code/config files."
        )

    # extract the raw text
    raw_text = ingestor.extract_text(filepath)

    if not raw_text or not raw_text.strip():
        print(f"warning: no text extracted from {path.name}")
        return []

    # chunk it
    source_name = path.name
    chunks = chunk_text(raw_text, chunk_size=chunk_size, overlap=overlap, source=source_name)

    # add the format info to each chunk
    format_name = _get_format_name(ext)
    for chunk in chunks:
        chunk["format"] = format_name

    return chunks


def extract_raw_text(filepath: str) -> str:
    """
    just extract text without chunking.
    useful if you want to do your own chunking strategy.
    """
    ingestor = _find_ingestor(filepath)
    if ingestor is None:
        raise ValueError(f"unsupported file format: {Path(filepath).suffix}")
    return ingestor.extract_text(filepath)


def is_supported(filepath: str) -> bool:
    """check if we can read this file type."""
    return _find_ingestor(filepath) is not None


def list_supported_formats() -> list:
    """return a list of supported file extensions."""
    return [
        ".pdf", ".docx",
        ".txt", ".md", ".rst", ".log",
        ".html", ".htm", ".xml",
        ".csv", ".tsv", ".json", ".jsonl",
        ".yaml", ".yml", ".toml", ".ini",
        ".py", ".js", ".ts", ".java", ".c", ".cpp",
        ".go", ".rs", ".rb", ".sh", ".sql",
        "...and most other text-based formats",
    ]


def _find_ingestor(filepath: str):
    """pick the right ingestor for a file."""
    for ing in _INGESTORS:
        if ing.can_handle(filepath):
            return ing
    return None


def _get_format_name(ext: str) -> str:
    """human readable format name for metadata."""
    names = {
        ".pdf": "pdf",
        ".docx": "docx",
        ".txt": "text",
        ".md": "markdown",
        ".html": "html",
        ".htm": "html",
        ".csv": "csv",
        ".tsv": "tsv",
        ".json": "json",
        ".jsonl": "jsonl",
        ".xml": "xml",
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
    }
    return names.get(ext, "text")
