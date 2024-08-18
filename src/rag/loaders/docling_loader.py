"""Docling document loader — PDF, DOCX, PPTX, HTML parsing with table extraction."""

from __future__ import annotations

import asyncio
from pathlib import Path

from langchain_core.documents import Document

from src.utils.logger import get_logger

log = get_logger(__name__)


async def load_document(file_path: str | Path) -> list[Document]:
    """Parse a document with Docling and return LangChain Documents.

    Uses asyncio.to_thread because Docling's converter is CPU-bound.
    Supports: PDF, DOCX, PPTX, HTML, images, Markdown.
    """
    from docling.document_converter import DocumentConverter

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    log.info("Parsing with Docling: %s", file_path.name)

    converter = DocumentConverter()
    result = await asyncio.to_thread(converter.convert, str(file_path))

    markdown_text = result.document.export_to_markdown()

    if not markdown_text.strip():
        log.warning("Docling produced empty output for %s", file_path.name)
        return []

    doc = Document(
        page_content=markdown_text,
        metadata={
            "source": str(file_path),
            "filename": file_path.name,
            "parser": "docling",
        },
    )

    return [doc]


async def load_documents(files: list[str | Path]) -> list[Document]:
    """Parse multiple files with Docling in parallel."""
    results = await asyncio.gather(*[load_document(f) for f in files], return_exceptions=True)

    docs = []
    for result in results:
        if isinstance(result, Exception):
            log.error("Docling error: %s", result)
        else:
            docs.extend(result)

    return docs
