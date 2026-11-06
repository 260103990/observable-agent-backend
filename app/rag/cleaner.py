"""Conservative normalization; never mutate the PDF reader's extracted text."""

from app.rag.models import CleanedPage, PdfPage


def clean_page(page: PdfPage) -> CleanedPage:
    """Preserve line structure, values and links, with exact source provenance.

    No dehyphenation, URL stripping or Unicode compatibility normalization.
    A collapsed whitespace character maps to its first retained source position.
    """
    if page.page_number < 1:
        raise ValueError("page_number must be a positive PDF page number")

    chars: list[str] = []
    offsets: list[int] = []
    raw = page.text
    index = 0
    while index < len(raw):
        char = raw[index]
        source_index = index
        index += 1
        if char == "\r":
            if index < len(raw) and raw[index] == "\n":
                index += 1
            char = "\n"

        if char == "\n":
            if chars and chars[-1] == " ":
                chars.pop()
                offsets.pop()
            if not chars or chars[-2:] == ["\n", "\n"]:
                continue
        elif char.isspace():
            if not chars or chars[-1] in (" ", "\n"):
                continue
            char = " "
        elif ord(char) < 32 or 127 <= ord(char) <= 159:
            continue

        chars.append(char)
        offsets.append(source_index)

    while chars and chars[-1].isspace():
        chars.pop()
        offsets.pop()

    return CleanedPage(
        page_number=page.page_number,
        raw_text=raw,
        text="".join(chars),
        source_offsets=tuple(offsets),
    )
