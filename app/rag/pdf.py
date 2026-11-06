from pathlib import Path

import pymupdf

from app.rag.models import PdfPage


class PdfReadError(Exception):
    pass


def _extract_obvious_columns(page: pymupdf.Page) -> str | None:
    words = page.get_text("words")

    if page.rotation != 0 or len(words) < 4:
        return None

    intervals = sorted((word[0], word[2]) for word in words)
    right_edge = intervals[0][1]
    gap_width = 0.0
    split_x = 0.0

    for x0, x1 in intervals[1:]:
        if x0 - right_edge > gap_width:
            gap_width = x0 - right_edge
            split_x = (right_edge + x0) / 2

        right_edge = max(right_edge, x1)

    if gap_width < page.rect.width * 0.05:
        return None

    left_words = [word for word in words if word[2] <= split_x]
    right_words = [word for word in words if word[0] >= split_x]

    if min(len(left_words), len(right_words)) < 2:
        return None

    bounds = []

    for column in (left_words, right_words):
        top = min(word[1] for word in column)
        bottom = max(word[3] for word in column)
        tallest_word = max(word[3] - word[1] for word in column)

        if bottom - top <= tallest_word * 1.5:
            return None

        bounds.append((top, bottom))

    if min(bounds[0][1], bounds[1][1]) <= max(
        bounds[0][0], bounds[1][0]
    ):
        return None

    rect = page.rect
    left_text = page.get_text(
        "text",
        clip=(rect.x0, rect.y0, split_x, rect.y1),
        sort=True,
    )
    right_text = page.get_text(
        "text",
        clip=(split_x, rect.y0, rect.x1, rect.y1),
        sort=True,
    )

    return left_text.rstrip() + "\n\n" + right_text.lstrip()


def extract_pdf_pages(path: Path) -> list[PdfPage]:
    pages: list[PdfPage] = []

    try:
        content = path.read_bytes()

        with pymupdf.open(stream=content, filetype="pdf") as doc:
            if doc.is_encrypted:
                raise PdfReadError(
                    "Password-protected PDFs are not supported"
                )

            if doc.page_count == 0:
                raise PdfReadError("PDF contains no pages")

            results = doc.to_text(
                page_chunks=True,
                use_ocr=False,
            )

            for item in results:
                text = item["text"]

                has_table = any(
                    box["class"] == "table"
                    for box in item["page_boxes"]
                )

                if not has_table:
                    page_number = item["metadata"]["page_number"]
                    column_text = _extract_obvious_columns(doc[page_number - 1])

                    if column_text is not None:
                        text = column_text

                if not text.strip():
                    text = ""

                pages.append(
                    PdfPage(
                        page_number=item["metadata"]["page_number"],
                        text=text,
                    )
                )

    except (OSError, pymupdf.FileDataError) as exc:
        raise PdfReadError(
            f"Failed to read PDF: {path.name}"
        ) from exc

    return pages
