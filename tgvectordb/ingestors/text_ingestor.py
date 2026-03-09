"""
text file ingestor.

handles all the "its basically just text" formats:
  .txt, .md, .rst, .log, .csv, .tsv, .json, .jsonl,
  .html, .xml, .yaml, .yml, .toml, .ini, .cfg,
  .py, .js, .ts, .java, .c, .cpp, .go, .rs, .rb, .sh, .sql
  ... and pretty much anything else thats utf-8 readable.

for csv/tsv it tries to be a bit smarter and turns rows into
readable text instead of raw comma separated garbage.

for html it strips the tags so you get clean text.

for code files it preserves structure but adds the filename
as context so the embeddings know what language its in.
"""

import re
import csv
import io
from pathlib import Path
from typing import Optional


# files where we just read them straight up
PLAIN_TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".text",
    ".log", ".ini", ".cfg", ".conf", ".toml",
    ".yaml", ".yml", ".env", ".properties",
}

# code files - we'll add filename context
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".c", ".cpp", ".h", ".hpp",
    ".go", ".rs", ".rb", ".php", ".swift",
    ".kt", ".scala", ".sh", ".bash", ".zsh",
    ".sql", ".r", ".m", ".lua", ".pl",
    ".cs", ".vb", ".dart", ".ex", ".exs",
    ".hs", ".clj", ".lisp", ".vim",
}

# structured data files
DATA_EXTENSIONS = {".csv", ".tsv", ".jsonl", ".ndjson"}


def can_handle(filepath: str) -> bool:
    """check if this ingestor knows how to read the file."""
    ext = Path(filepath).suffix.lower()
    all_supported = PLAIN_TEXT_EXTENSIONS | CODE_EXTENSIONS | DATA_EXTENSIONS | {".html", ".htm", ".xml", ".json"}
    return ext in all_supported


def extract_text(filepath: str) -> str:
    """
    read a text-based file and return clean text content.
    picks the right strategy based on file extension.
    """
    path = Path(filepath)
    ext = path.suffix.lower()

    if not path.exists():
        raise FileNotFoundError(f"file not found: {filepath}")

    raw = path.read_text(encoding="utf-8", errors="replace")

    if ext in (".html", ".htm"):
        return _strip_html(raw)

    elif ext == ".xml":
        return _strip_xml(raw)

    elif ext == ".csv":
        return _csv_to_text(raw, delimiter=",")

    elif ext == ".tsv":
        return _csv_to_text(raw, delimiter="\t")

    elif ext in (".jsonl", ".ndjson"):
        return _jsonl_to_text(raw)

    elif ext == ".json":
        return _json_to_text(raw)

    elif ext in CODE_EXTENSIONS:
        return _code_to_text(raw, path.name)

    else:
        # plain text, just return as is
        return raw


def _strip_html(html_content: str) -> str:
    """
    remove html tags and get the readable text out.
    not using beautifulsoup to avoid the dependency - regex is fine for this.
    """
    # remove script and style blocks entirely
    text = re.sub(r"<script[^>]*>.*?</script>", " ", html_content, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)

    # remove html comments
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)

    # add newlines for block elements so paragraphs are preserved
    block_tags = r"</?(p|div|br|hr|h[1-6]|li|tr|blockquote|pre|section|article|header|footer)[^>]*>"
    text = re.sub(block_tags, "\n", text, flags=re.IGNORECASE)

    # strip remaining tags
    text = re.sub(r"<[^>]+>", " ", text)

    # decode common html entities
    text = text.replace("&nbsp;", " ")
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")

    # clean up whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _strip_xml(xml_content: str) -> str:
    """similar to html stripping but less aggressive."""
    text = re.sub(r"<[^>]+>", " ", xml_content)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _csv_to_text(csv_content: str, delimiter: str = ",") -> str:
    """
    turn csv rows into readable text.
    uses the header row as field names so each row becomes like:
    "Name: John, Age: 25, City: Mumbai"
    way better for embeddings than raw csv.
    """
    reader = csv.reader(io.StringIO(csv_content), delimiter=delimiter)
    rows = list(reader)

    if len(rows) < 2:
        # just one row or empty, return as plain text
        return csv_content

    headers = [h.strip() for h in rows[0]]
    text_rows = []

    for row in rows[1:]:
        if not any(cell.strip() for cell in row):
            continue  # skip empty rows

        parts = []
        for i, cell in enumerate(row):
            cell = cell.strip()
            if not cell:
                continue
            if i < len(headers) and headers[i]:
                parts.append(f"{headers[i]}: {cell}")
            else:
                parts.append(cell)

        if parts:
            text_rows.append(". ".join(parts))

    return "\n\n".join(text_rows)


def _jsonl_to_text(content: str) -> str:
    """turn jsonl (one json object per line) into readable text."""
    import json

    text_parts = []
    for line_num, line in enumerate(content.strip().split("\n")):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            # try to find a text-like field
            text = _extract_text_from_json_obj(obj)
            if text:
                text_parts.append(text)
        except json.JSONDecodeError:
            # not valid json, just include the raw line
            text_parts.append(line)

    return "\n\n".join(text_parts)


def _json_to_text(content: str) -> str:
    """
    try to extract readable text from a json file.
    handles both single objects and arrays.
    """
    import json

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # not valid json, just return raw
        return content

    if isinstance(data, list):
        parts = []
        for item in data:
            if isinstance(item, dict):
                text = _extract_text_from_json_obj(item)
                if text:
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n\n".join(parts)

    elif isinstance(data, dict):
        return _extract_text_from_json_obj(data)

    else:
        return str(data)


def _extract_text_from_json_obj(obj: dict) -> str:
    """
    pull readable text out of a json object.
    looks for common field names like text, content, body, description, etc.
    if none found, just dumps key: value pairs.
    """
    # fields that are most likely to contain the "main" text
    text_field_names = [
        "text", "content", "body", "description", "message",
        "summary", "title", "name", "question", "answer",
        "comment", "note", "abstract", "snippet",
    ]

    # first check if theres a dedicated text field
    for field in text_field_names:
        if field in obj and isinstance(obj[field], str) and len(obj[field]) > 10:
            # also grab other short fields as context
            context_parts = []
            for k, v in obj.items():
                if k == field:
                    continue
                if isinstance(v, str) and len(v) < 200:
                    context_parts.append(f"{k}: {v}")
            context = ". ".join(context_parts[:5])
            main_text = obj[field]
            if context:
                return f"{context}\n{main_text}"
            return main_text

    # no obvious text field, just dump everything readable
    parts = []
    for k, v in obj.items():
        if isinstance(v, str):
            parts.append(f"{k}: {v}")
        elif isinstance(v, (int, float, bool)):
            parts.append(f"{k}: {v}")
    return ". ".join(parts)


def _code_to_text(code_content: str, filename: str) -> str:
    """
    prepare code for embedding.
    adds filename as context and cleans up a bit,
    but preserves the actual code structure because
    code semantics matter for search.
    """
    # add filename as context - helps the embedding model
    # understand what language / what this file is about
    header = f"File: {filename}\n\n"

    # strip very long comment blocks at the top (license headers etc)
    lines = code_content.split("\n")
    cleaned_lines = []
    skipping_header_comments = True

    for line in lines:
        stripped = line.strip()

        if skipping_header_comments:
            # skip empty lines and comments at the very top
            if not stripped:
                continue
            if stripped.startswith(("#!", "//", "/*", " *", "* ", "*/")):
                continue
            if stripped.startswith("#") and len(stripped) > 1:
                # could be a python comment or a shebang, check if its a license block
                # heuristic: if first 20 lines are all comments, skip them
                continue
            skipping_header_comments = False

        cleaned_lines.append(line)

    # rejoin, but cap it if the file is absurdly long
    # (we'll chunk it anyway, but no point processing a 50k line file as one blob)
    code_text = "\n".join(cleaned_lines)

    return header + code_text
