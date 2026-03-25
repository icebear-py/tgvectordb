import csv
import io
import re
from pathlib import Path

PLAIN_TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".text",
    ".log",
    ".ini",
    ".cfg",
    ".conf",
    ".toml",
    ".yaml",
    ".yml",
    ".env",
    ".properties",
}
CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".scala",
    ".sh",
    ".bash",
    ".zsh",
    ".sql",
    ".r",
    ".m",
    ".lua",
    ".pl",
    ".cs",
    ".vb",
    ".dart",
    ".ex",
    ".exs",
    ".hs",
    ".clj",
    ".lisp",
    ".vim",
}
DATA_EXTENSIONS = {".csv", ".tsv", ".jsonl", ".ndjson"}


def can_handle(filepath: str) -> bool:
    ext = Path(filepath).suffix.lower()
    all_supported = (
        PLAIN_TEXT_EXTENSIONS
        | CODE_EXTENSIONS
        | DATA_EXTENSIONS
        | {".html", ".htm", ".xml", ".json"}
    )
    return ext in all_supported


def extract_text(filepath: str) -> str:
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
        return raw


def _strip_html(html_content: str) -> str:
    text = re.sub(
        r"<script[^>]*>.*?</script>", " ", html_content, flags=re.DOTALL | re.IGNORECASE
    )
    text = re.sub(
        r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE
    )
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    block_tags = r"</?(p|div|br|hr|h[1-6]|li|tr|blockquote|pre|section|article|header|footer)[^>]*>"
    text = re.sub(block_tags, "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&nbsp;", " ")
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_xml(xml_content: str) -> str:
    text = re.sub(r"<[^>]+>", " ", xml_content)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _csv_to_text(csv_content: str, delimiter: str = ",") -> str:
    reader = csv.reader(io.StringIO(csv_content), delimiter=delimiter)
    rows = list(reader)
    if len(rows) < 2:
        return csv_content
    headers = [h.strip() for h in rows[0]]
    text_rows = []
    for row in rows[1:]:
        if not any(cell.strip() for cell in row):
            continue
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
    import json

    text_parts = []
    for line_num, line in enumerate(content.strip().split("\n")):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            text = _extract_text_from_json_obj(obj)
            if text:
                text_parts.append(text)
        except json.JSONDecodeError:
            text_parts.append(line)
    return "\n\n".join(text_parts)


def _json_to_text(content: str) -> str:
    import json

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
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
    text_field_names = [
        "text",
        "content",
        "body",
        "description",
        "message",
        "summary",
        "title",
        "name",
        "question",
        "answer",
        "comment",
        "note",
        "abstract",
        "snippet",
    ]
    for field in text_field_names:
        if field in obj and isinstance(obj[field], str) and len(obj[field]) > 10:
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
    parts = []
    for k, v in obj.items():
        if isinstance(v, str):
            parts.append(f"{k}: {v}")
        elif isinstance(v, (int, float, bool)):
            parts.append(f"{k}: {v}")
    return ". ".join(parts)


def _code_to_text(code_content: str, filename: str) -> str:
    header = f"File: {filename}\n\n"
    lines = code_content.split("\n")
    cleaned_lines = []
    skipping_header_comments = True
    for line in lines:
        stripped = line.strip()
        if skipping_header_comments:
            if not stripped:
                continue
            if stripped.startswith(("#!", "//", "/*", " *", "* ", "*/")):
                continue
            if stripped.startswith("#") and len(stripped) > 1:
                continue
            skipping_header_comments = False
        cleaned_lines.append(line)
    code_text = "\n".join(cleaned_lines)
    return header + code_text
