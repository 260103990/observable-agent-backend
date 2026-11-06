"""The pure-text portion of ingestion; no model, database or HTTP ownership."""

from app.rag.cleaner import clean_page
from app.rag.models import PdfPage, PreparedText
from app.rag.splitter import (
    DEFAULT_CHUNK_SIZE_CHARS,
    DEFAULT_OVERLAP_CHARS,
    split_page,
)


class NoUsableTextError(ValueError):
    """The first version requires a PDF with extractable text, without OCR."""


def prepare_pages(
    pages: list[PdfPage],
    *,
    chunk_size_chars: int = DEFAULT_CHUNK_SIZE_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> PreparedText:
    """Retain blank pages and their numbers, but produce only useful chunks."""
    previous_number = 0
    cleaned_pages = []
    chunks = []
    for page in pages:
        if page.page_number <= previous_number:
            raise ValueError("PDF page numbers must be positive and strictly increasing")
        previous_number = page.page_number
        cleaned = clean_page(page)
        cleaned_pages.append(cleaned)
        chunks.extend(split_page(
            cleaned,
            chunk_size_chars=chunk_size_chars,
            overlap_chars=overlap_chars,
        ))
    if not chunks:
        raise NoUsableTextError("PDF contains no usable text; a text PDF is required")
    return PreparedText(pages=cleaned_pages, chunks=chunks)
