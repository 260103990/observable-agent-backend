import pytest

from app.rag.models import PdfPage


def test_real_pdf_reader_to_chunks_preserves_blank_pages_and_page_sources(tmp_path):
    from app.rag.pdf import extract_pdf_pages
    from app.rag.preparation import prepare_pages
    from tests.pdf_helpers import write_text_pdf

    path = tmp_path / "paper.pdf"
    write_text_pdf(path, ["Paper cover", None, "Gain: 42 req/s. Hit rate: 85%."])
    source = extract_pdf_pages(path)
    result = prepare_pages(source, chunk_size_chars=22, overlap_chars=4)
    assert [p.page_number for p in result.pages] == [1, 2, 3]
    assert [p.raw_text for p in result.pages] == [p.text for p in source]
    assert result.pages[1].text == ""
    assert {c.page_number for c in result.chunks} == {1, 3}
    for chunk in result.chunks:
        page = result.pages[chunk.page_number - 1]
        assert chunk.raw_text == page.raw_text[chunk.raw_start_char:chunk.raw_end_char]
        assert chunk.text == page.text[chunk.start_char:chunk.end_char]


@pytest.mark.parametrize("pages", [[], [PdfPage(page_number=1, text=" \n\t")]])
def test_preparation_rejects_document_without_usable_text(pages):
    from app.rag.preparation import NoUsableTextError, prepare_pages

    with pytest.raises(NoUsableTextError, match="text"):
        prepare_pages(pages)


@pytest.mark.parametrize("numbers", [[1, 1], [3, 2], [0], [-1]])
def test_preparation_rejects_ambiguous_or_invalid_page_order(numbers):
    from app.rag.preparation import prepare_pages

    with pytest.raises(ValueError, match="page"):
        prepare_pages([PdfPage(page_number=n, text="content") for n in numbers])


def test_preparation_does_not_renumber_noncontiguous_source_pages():
    from app.rag.preparation import prepare_pages

    result = prepare_pages([PdfPage(page_number=2, text="A"), PdfPage(page_number=7, text="B")])
    assert [p.page_number for p in result.pages] == [2, 7]
    assert [c.page_number for c in result.chunks] == [2, 7]
