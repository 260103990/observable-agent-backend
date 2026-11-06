import pytest

from app.rag.models import PdfPage


def split(raw, *, size=30, overlap=5, page_number=4):
    from app.rag.cleaner import clean_page
    from app.rag.splitter import split_page

    page = clean_page(PdfPage(page_number=page_number, text=raw))
    return page, split_page(page, chunk_size_chars=size, overlap_chars=overlap)


def assert_provenance(page, chunks, size):
    covered = set()
    for index, chunk in enumerate(chunks):
        assert chunk.page_number == page.page_number
        assert chunk.chunk_index == index
        assert chunk.text == page.text[chunk.start_char:chunk.end_char]
        assert chunk.raw_text == page.raw_text[chunk.raw_start_char:chunk.raw_end_char]
        assert chunk.raw_start_char == page.source_offsets[chunk.start_char]
        assert chunk.raw_end_char == page.source_offsets[chunk.end_char - 1] + 1
        assert 0 < len(chunk.text) <= size
        assert chunk.text.strip() == chunk.text
        covered.update(range(chunk.start_char, chunk.end_char))
    assert all(i in covered for i, char in enumerate(page.text) if not char.isspace())
    assert all(a.start_char < b.start_char for a, b in zip(chunks, chunks[1:]))
    assert all(a.end_char < b.end_char for a, b in zip(chunks, chunks[1:]))


def test_short_paragraphs_merge_and_keep_original_source():
    page, chunks = split(" \tFirst paragraph.\r\n\r\nSecond paragraph.\x00 ", size=80)
    assert [c.text for c in chunks] == ["First paragraph.\n\nSecond paragraph."]
    assert_provenance(page, chunks, 80)


def test_prefers_paragraph_boundary_over_cutting_next_paragraph():
    first = "A" * 24
    page, chunks = split(first + "\n\n" + "B" * 30, size=40, overlap=0)
    assert chunks[0].text == first
    assert chunks[1].text == "B" * 30
    assert_provenance(page, chunks, 40)


def test_prefers_sentence_boundary_within_long_paragraph():
    page, chunks = split("A" * 20 + ". " + "B" * 30, size=40, overlap=0)
    assert chunks[0].text == "A" * 20 + "."
    assert_provenance(page, chunks, 40)


@pytest.mark.parametrize("raw", ["0123456789" * 20, "中文检索与缓存实验" * 20])
@pytest.mark.parametrize("overlap", [0, 4, 29])
def test_long_unbroken_text_is_bounded_and_fully_covered(raw, overlap):
    page, chunks = split(raw, overlap=overlap)
    assert chunks[0].start_char == 0
    assert chunks[-1].end_char == len(page.text)
    assert_provenance(page, chunks, 30)
    for left, right in zip(chunks, chunks[1:]):
        assert left.end_char - right.start_char == overlap


def test_split_keeps_numbers_symbols_urls_and_whole_text():
    page, chunks = split(
        "\tThroughput:\t42 req/s.\r\n\r\nHit rate: 85%. "
        "Temperature: -3.5 °C. URL: https://example.org/a-b", size=32, overlap=7,
    )
    assert_provenance(page, chunks, 32)


def test_empty_page_produces_no_chunk():
    _, chunks = split(" \r\n\t")
    assert chunks == []


@pytest.mark.parametrize("separator", [" ", "\n\n"])
def test_high_overlap_advances_end_when_window_ends_in_whitespace(separator):
    page, chunks = split("A" * 30 + separator + "B" * 30, overlap=29)
    assert_provenance(page, chunks, 30)


@pytest.mark.parametrize("punctuation", ["。", "！", "？"])
def test_chinese_sentence_boundary_needs_no_following_space(punctuation):
    page, chunks = split("甲" * 24 + punctuation + "乙" * 35, size=40, overlap=0)
    assert chunks[0].text == "甲" * 24 + punctuation
    assert_provenance(page, chunks, 40)


@pytest.mark.parametrize("size,overlap", [
    (0, 0), (-1, 0), (5, -1), (5, 5), (5, 6),
    (True, 0), (5, False), (5.5, 0), (5, 1.5),
])
def test_invalid_chunk_configuration_is_rejected_even_for_empty_page(size, overlap):
    from app.rag.cleaner import clean_page
    from app.rag.splitter import split_page

    with pytest.raises(ValueError):
        split_page(clean_page(PdfPage(page_number=1, text="")),
                   chunk_size_chars=size, overlap_chars=overlap)


def test_character_unit_is_unicode_characters_and_output_is_repeatable():
    from app.rag.splitter import split_page

    page, chunks = split("研究缓存🙂" * 10, size=9, overlap=2)
    assert len(chunks[0].text) == 9
    assert len(chunks[0].text.encode("utf-8")) > 9
    assert chunks == split_page(page, chunk_size_chars=9, overlap_chars=2)
    assert_provenance(page, chunks, 9)
