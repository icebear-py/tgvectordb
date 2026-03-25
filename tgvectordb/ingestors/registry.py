from pathlib import Path

from tgvectordb.embedding.chunker import chunk_text
from tgvectordb.ingestors import docx_ingestor, pdf_ingestor, text_ingestor

_INGESTORS = [
    pdf_ingestor,
    docx_ingestor,
    text_ingestor,
]


def ingest(
    filepath: str,
    chunk_size: int = 400,
    overlap: int = 50,
) -> list:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"cant find: {filepath}")
    ext = path.suffix.lower()
    ingestor = _find_ingestor(filepath)
    if ingestor is None:
        raise ValueError(
            f"dont know how to read '{ext}' files.\n"
            f"supported: .pdf, .docx, .txt, .md, .html, .csv, .json, "
            f"and most code/config files."
        )
    raw_text = ingestor.extract_text(filepath)
    if not raw_text or not raw_text.strip():
        print(f"warning: no text extracted from {path.name}")
        return []
    source_name = path.name
    chunks = chunk_text(
        raw_text, chunk_size=chunk_size, overlap=overlap, source=source_name
    )
    format_name = _get_format_name(ext)
    for chunk in chunks:
        chunk["format"] = format_name
    return chunks


def extract_raw_text(filepath: str) -> str:
    ingestor = _find_ingestor(filepath)
    if ingestor is None:
        raise ValueError(f"unsupported file format: {Path(filepath).suffix}")
    return ingestor.extract_text(filepath)


def is_supported(filepath: str) -> bool:
    return _find_ingestor(filepath) is not None


def list_supported_formats() -> list:
    return [
        ".pdf",
        ".docx",
        ".txt",
        ".md",
        ".rst",
        ".log",
        ".html",
        ".htm",
        ".xml",
        ".csv",
        ".tsv",
        ".json",
        ".jsonl",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".py",
        ".js",
        ".ts",
        ".java",
        ".c",
        ".cpp",
        ".go",
        ".rs",
        ".rb",
        ".sh",
        ".sql",
        "...and most other text-based formats",
    ]


def _find_ingestor(filepath: str):
    for ing in _INGESTORS:
        if ing.can_handle(filepath):
            return ing
    return None


def _get_format_name(ext: str) -> str:
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
