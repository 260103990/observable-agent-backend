from pypdf import PdfWriter

import pytest


def test_extract_pdf_pages_preserves_blank_page_numbers(tmp_path):
    pdf_path = tmp_path / "three_blank_pages.pdf"
    writer = PdfWriter()
    for _ in range(3):
        writer.add_blank_page(width=612, height=792)
    writer.write(pdf_path)

    from app.rag.pdf import extract_pdf_pages

    pages = extract_pdf_pages(pdf_path)

    assert [page.page_number for page in pages] == [1, 2, 3]
    assert [page.text for page in pages] == ["", "", ""]


def test_extract_pdf_pages_keeps_text_on_original_pages(tmp_path):
    from app.rag.pdf import extract_pdf_pages
    from tests.pdf_helpers import write_text_pdf

    pdf_path = tmp_path / "paper.pdf"
    write_text_pdf(pdf_path, [
        "Paper cover",
        None,
        "Baseline: no cache. Throughput: 42 req/s. Hit rate: 85%.",
    ])

    pages = extract_pdf_pages(pdf_path)

    assert [page.page_number for page in pages] == [1, 2, 3]
    assert pages[0].text.strip() == "Paper cover"
    assert pages[1].text == ""
    assert pages[2].text.strip() == (
        "Baseline: no cache. Throughput: 42 req/s. Hit rate: 85%."
    )


@pytest.mark.parametrize("content", [b"", b"not a pdf"])
def test_extract_pdf_pages_rejects_invalid_pdf(tmp_path, content):
    from app.rag.pdf import PdfReadError, extract_pdf_pages

    pdf_path = tmp_path / "invalid.pdf"
    pdf_path.write_bytes(content)

    with pytest.raises(PdfReadError):
        extract_pdf_pages(pdf_path)


def test_extract_pdf_pages_wraps_missing_file(tmp_path):
    from app.rag.pdf import PdfReadError, extract_pdf_pages

    with pytest.raises(PdfReadError) as caught:
        extract_pdf_pages(tmp_path / "missing.pdf")

    assert isinstance(caught.value.__cause__, FileNotFoundError)


def test_extract_pdf_pages_rejects_encrypted_pdf(tmp_path):
    from app.rag.pdf import PdfReadError, extract_pdf_pages

    pdf_path = tmp_path / "protected.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.encrypt("secret")
    writer.write(pdf_path)

    with pytest.raises(PdfReadError):
        extract_pdf_pages(pdf_path)


def test_extract_pdf_pages_rejects_document_without_pages(tmp_path):
    from app.rag.pdf import PdfReadError, extract_pdf_pages
    from tests.pdf_helpers import write_text_pdf

    pdf_path = tmp_path / "no_pages.pdf"
    write_text_pdf(pdf_path, [])

    with pytest.raises(PdfReadError):
        extract_pdf_pages(pdf_path)


@pytest.mark.parametrize(
    "blocks, expected_order",
    [
        (
            [
                (72, 100, "Footer"),
                (72, 700, "Title"),
                (72, 500, "Body"),
            ],
            ["Title", "Body", "Footer"],
        ),
        (
            [
                (72, 700, "Left-top"),
                (330, 700, "Right-top"),
                (72, 650, "Left-bottom"),
                (330, 650, "Right-bottom"),
            ],
            ["Left-top", "Left-bottom", "Right-top", "Right-bottom"],
        ),
    ],
    ids=["single-column", "two-column"],
)
def test_extract_pdf_pages_follows_reading_order(
    tmp_path, blocks, expected_order
):
    from app.rag.pdf import extract_pdf_pages
    from tests.pdf_helpers import write_text_pdf

    template_path = tmp_path / "template.pdf"
    write_text_pdf(template_path, ["placeholder"])

    writer = PdfWriter(clone_from=template_path)
    content = "".join(
        f"BT /F1 12 Tf {x} {y} Td ({text}) Tj ET\n"
        for x, y, text in blocks
    ).encode("ascii")
    writer.pages[0]["/Contents"].set_data(content)

    pdf_path = tmp_path / "reading_order.pdf"
    writer.write(pdf_path)

    text = extract_pdf_pages(pdf_path)[0].text
    positions = [text.index(item) for item in expected_order]

    assert positions == sorted(positions)
