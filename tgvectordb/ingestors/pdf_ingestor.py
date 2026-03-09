"""
pdf ingestor - extracts text from PDF files.

uses pdfplumber which is pretty good at preserving layout
and can also pull out tables as structured data.

for scanned PDFs (images of text), this wont work great -
you'd need OCR (tesseract) which we dont include by default
because its a pain to install. maybe in v2.

needs pdfplumber: pip install pdfplumber
  or: pip install tgvectordb[pdf]
"""

from pathlib import Path


def can_handle(filepath: str) -> bool:
    ext = Path(filepath).suffix.lower()
    return ext == ".pdf"


def extract_text(filepath: str, include_tables: bool = True) -> str:
    """
    extract text from a PDF file.
    goes page by page, grabs regular text and optionally tables.
    adds page markers so the chunker can track where text came from.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"file not found: {filepath}")

    try:
        import pdfplumber
    except ImportError:
        raise ImportError(
            "pdfplumber is needed for PDF support. install with:\n"
            "  pip install pdfplumber\n"
            "  or: pip install tgvectordb[pdf]"
        )

    all_text = []

    with pdfplumber.open(str(path)) as pdf:
        total_pages = len(pdf.pages)

        for page_num, page in enumerate(pdf.pages):
            page_parts = []

            # get the main text
            page_text = page.extract_text()
            if page_text and page_text.strip():
                page_parts.append(page_text.strip())

            # try to get tables separately - sometimes pdfplumber
            # does a better job with tables when you extract them explicitly
            if include_tables:
                tables = page.extract_tables()
                for table in tables:
                    table_text = _table_to_text(table)
                    if table_text:
                        # only add if its not already in the main text
                        # (pdfplumber sometimes includes table text in both)
                        if table_text not in (page_text or ""):
                            page_parts.append(f"Table:\n{table_text}")

            if page_parts:
                # add a subtle page marker - useful for metadata
                page_header = f"[Page {page_num + 1}/{total_pages}]"
                page_content = "\n".join(page_parts)
                all_text.append(f"{page_header}\n{page_content}")

    return "\n\n".join(all_text)


def _table_to_text(table: list) -> str:
    """
    convert a pdfplumber table (list of lists) to readable text.
    same idea as the csv ingestor - use first row as headers if it looks right.
    """
    if not table or len(table) < 1:
        return ""

    # clean up None values
    cleaned = []
    for row in table:
        cleaned_row = [(cell or "").strip() for cell in row]
        cleaned.append(cleaned_row)

    headers = cleaned[0]
    looks_like_headers = all(
        len(h) < 50 and not h.replace(".", "").replace(",", "").isdigit()
        for h in headers if h
    )

    rows_text = []

    if looks_like_headers and len(cleaned) > 1:
        for row in cleaned[1:]:
            parts = []
            for i, cell in enumerate(row):
                if not cell:
                    continue
                if i < len(headers) and headers[i]:
                    parts.append(f"{headers[i]}: {cell}")
                else:
                    parts.append(cell)
            if parts:
                rows_text.append(". ".join(parts))
    else:
        for row in cleaned:
            row_str = " | ".join(c for c in row if c)
            if row_str:
                rows_text.append(row_str)

    return "\n".join(rows_text)
