import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from pypdf import PdfReader


def _extract_with_docling(file_path):
    try:
        from docling.document_converter import DocumentConverter
    except Exception:
        return None

    try:
        converter = DocumentConverter()
        result = converter.convert(str(file_path))
        markdown = result.document.export_to_markdown()
        if markdown and markdown.strip():
            return markdown
    except Exception:
        return None

    return None


def _extract_with_marker_cli(file_path):
    marker_binary = shutil.which("marker_single") or shutil.which("marker")
    if not marker_binary:
        return None

    commands = [
        [marker_binary, str(file_path), "{out}", "--output_format", "markdown"],
        [marker_binary, "--output_format", "markdown", str(file_path), "{out}"],
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        for command in commands:
            expanded = [arg.format(out=str(out_dir)) for arg in command]
            try:
                subprocess.run(
                    expanded,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=180,
                )
            except Exception:
                continue

            markdown_files = sorted(out_dir.rglob("*.md"))
            if not markdown_files:
                continue

            parts = []
            for md_file in markdown_files:
                try:
                    content = md_file.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                if content.strip():
                    parts.append(content)

            if parts:
                return "\n\n".join(parts)

    return None


def _extract_with_pypdf(file_path):
    text = ""
    reader = PdfReader(str(file_path))
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text


def extract_pdf_content(file_path):
    """Extract PDF content with a markdown-first strategy.

    Parser order is controlled by PDF_PARSER env var: auto|docling|marker|pypdf.
    Returns a tuple of (text, parser_name).
    """

    parser_preference = os.environ.get("PDF_PARSER", "auto").strip().lower()

    if parser_preference not in {"auto", "docling", "marker", "pypdf"}:
        parser_preference = "auto"

    if parser_preference in {"auto", "docling"}:
        markdown = _extract_with_docling(file_path)
        if markdown:
            return markdown, "docling"
        if parser_preference == "docling":
            fallback = _extract_with_pypdf(file_path)
            return fallback, "pypdf"

    if parser_preference in {"auto", "marker"}:
        markdown = _extract_with_marker_cli(file_path)
        if markdown:
            return markdown, "marker"
        if parser_preference == "marker":
            fallback = _extract_with_pypdf(file_path)
            return fallback, "pypdf"

    return _extract_with_pypdf(file_path), "pypdf"
