import pytest

from app.rag.models import PdfPage


def test_clean_page_preserves_original_and_maps_normalized_characters():
    from app.rag.cleaner import clean_page

    raw = "  Title\r\n\r\n  Gain:\t42\u00a0req/s.  \r\nNext line.\x00\r\n"
    result = clean_page(PdfPage(page_number=3, text=raw))

    assert result.page_number == 3
    assert result.raw_text == raw
    assert result.text == "Title\n\nGain: 42 req/s.\nNext line."
    assert len(result.source_offsets) == len(result.text)
    assert list(result.source_offsets) == sorted(set(result.source_offsets))
    for char, offset in zip(result.text, result.source_offsets):
        assert 0 <= offset < len(raw)
        if not char.isspace():
            assert char == raw[offset]


def test_clean_page_keeps_experimental_values_links_and_hyphens():
    from app.rag.cleaner import clean_page

    raw = (
        "-3.5 °C; 10–20 ms; 85%; p<0.05; x_1 = -2; state-of-the-art.\n"
        "https://example.org/a-b?q=1 user@example.org\n"
        "syn-\nthesize\n| A | B |\n| 1 | 2 |"
    )
    assert clean_page(PdfPage(page_number=1, text=raw)).text == raw


@pytest.mark.parametrize("raw", ["", " \t\r\n\u00a0\n", "\x00\x01\x7f"])
def test_clean_page_preserves_empty_page_record(raw):
    from app.rag.cleaner import clean_page

    result = clean_page(PdfPage(page_number=7, text=raw))
    assert (result.page_number, result.raw_text, result.text) == (7, raw, "")
    assert not result.source_offsets


def test_cleaning_is_idempotent_without_modifying_input():
    from app.rag.cleaner import clean_page

    page = PdfPage(page_number=1, text=" A  B\r\n \n\n C\tD\x00 ")
    snapshot = page.model_dump()
    first = clean_page(page)
    second = clean_page(PdfPage(page_number=1, text=first.text))
    assert first.text == "A B\n\nC D"
    assert second.text == first.text
    assert page.model_dump() == snapshot


@pytest.mark.parametrize("page_number", [0, -1])
def test_clean_page_rejects_invalid_source_page(page_number):
    from app.rag.cleaner import clean_page

    with pytest.raises(ValueError, match="page_number"):
        clean_page(PdfPage(page_number=page_number, text="text"))


@pytest.mark.parametrize("offsets", [(), (0, 1), (0, 1, 3), (0, 2, 1), (-1, 1, 2)])
def test_cleaned_page_rejects_invalid_source_mapping(offsets):
    from app.rag.models import CleanedPage

    with pytest.raises(ValueError, match="source_offsets"):
        CleanedPage(page_number=1, raw_text="abc", text="abc", source_offsets=offsets)
