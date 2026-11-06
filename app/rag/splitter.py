"""Single-page, character-counted splitting with original-text ranges."""

import re

from app.rag.models import CleanedPage, TextChunk


DEFAULT_CHUNK_SIZE_CHARS = 1200
DEFAULT_OVERLAP_CHARS = 150

# Preferred boundaries are chosen only in the latter half of a full window.
# These are local heuristics, not an implementation of Dify's splitter.
_BOUNDARIES = (
    re.compile(r"\n\n"),
    re.compile(r"[。！？]|[.!?](?=\s)"),
    re.compile(r"\n"),
    re.compile(r" "),
)


def split_page(
    page: CleanedPage,
    *,
    chunk_size_chars: int = DEFAULT_CHUNK_SIZE_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[TextChunk]:
    """Split contiguous normalized slices; overlap is a maximum target.

    Whitespace trimming may reduce overlap. Starts and ends strictly advance,
    including when overlap is close to size. No non-whitespace text is dropped.
    """
    if type(chunk_size_chars) is not int or chunk_size_chars <= 0:
        raise ValueError("chunk_size_chars must be a positive integer")
    if type(overlap_chars) is not int or not 0 <= overlap_chars < chunk_size_chars:
        raise ValueError("overlap_chars must satisfy 0 <= overlap < chunk_size_chars")

    text = page.text
    chunks: list[TextChunk] = []
    start = 0
    previous_end = 0
    while start < len(text):
        while start < len(text) and text[start].isspace():
            start += 1
        if start == len(text):
            break

        end = min(start + chunk_size_chars, len(text))
        if end < len(text):
            lower = max(start + max(1, chunk_size_chars // 2), previous_end + 1)
            for priority, pattern in enumerate(_BOUNDARIES):
                candidates = [
                    match.end() if priority == 1 else match.start()
                    for match in pattern.finditer(text, start, end)
                ]
                candidates = [position for position in candidates if position >= lower]
                if candidates:
                    end = candidates[-1]
                    break

        while end > start and text[end - 1].isspace():
            end -= 1

        # Hard-window whitespace trimming can otherwise create a chunk wholly
        # inside the previous one when overlap is close to the window length.
        if end <= previous_end:
            start += 1
            continue

        raw_start = page.source_offsets[start]
        raw_end = page.source_offsets[end - 1] + 1
        chunks.append(TextChunk(
            page_number=page.page_number,
            chunk_index=len(chunks),
            text=text[start:end],
            raw_text=page.raw_text[raw_start:raw_end],
            start_char=start,
            end_char=end,
            raw_start_char=raw_start,
            raw_end_char=raw_end,
        ))
        if end == len(text):
            break
        previous_end = end
        start = max(start + 1, end - overlap_chars)

    return chunks
