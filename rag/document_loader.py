"""Document loaders for TXT, PDF, DOCX, JSON, CSV, and Markdown files."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def load_document(path: str) -> list[dict]:
    """Load a document and return a list of page dicts.

    Each dict has keys ``content`` (str) and ``metadata`` (dict with at least
    ``source`` and ``page`` keys).
    """
    ext = Path(path).suffix.lower()
    loaders = {
        ".txt": _load_txt,
        ".md": _load_txt,
        ".pdf": _load_pdf,
        ".docx": _load_docx,
        ".json": _load_json,
        ".csv": _load_csv,
    }
    loader = loaders.get(ext)
    if loader is None:
        raise ValueError(f"Unsupported file format: {ext!r}")
    return loader(path)


# ---------------------------------------------------------------------------
# Format-specific loaders
# ---------------------------------------------------------------------------

def _load_txt(path: str) -> list[dict]:
    with open(path, encoding="utf-8", errors="replace") as f:
        content = f.read()
    return [{"content": content, "metadata": {"source": path, "page": 1}}]


def _load_pdf(path: str) -> list[dict]:
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(
                {"content": text, "metadata": {"source": path, "page": i}}
            )
    if not pages:
        import warnings
        warnings.warn(
            f"No extractable text found in PDF: {path!r}. "
            "The file may be scanned or image-only.",
            UserWarning,
            stacklevel=2,
        )
    return pages


def _load_docx(path: str) -> list[dict]:
    from docx import Document

    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    content = "\n\n".join(paragraphs)
    return [{"content": content, "metadata": {"source": path, "page": 1}}]


def _load_json(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        content = "\n".join(str(item) for item in data)
    elif isinstance(data, dict):
        content = "\n".join(f"{k}: {v}" for k, v in data.items())
    else:
        content = str(data)

    return [{"content": content, "metadata": {"source": path, "page": 1}}]


def _load_csv(path: str) -> list[dict]:
    rows: list[str] = []
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(str(dict(row)))
    content = "\n".join(rows)
    return [{"content": content, "metadata": {"source": path, "page": 1}}]
