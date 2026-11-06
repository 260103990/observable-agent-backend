import asyncio

import pytest

from app.rag.models import PdfPage
from app.rag.preparation import prepare_pages


def sample(paper_id="paper-1"):
    from app.rag.storage_models import EmbeddingSpec, PaperInfo

    prepared = prepare_pages([PdfPage(page_number=1, text="  Cache\x00 hit.\r\n\r\nResults: 42% 🙂。")],
                             chunk_size_chars=18, overlap_chars=2)
    info = PaperInfo(paper_id=paper_id, title="Paper", source_path="paper.pdf", source_sha256="a" * 64,
                     embedding=EmbeddingSpec(model="qwen3-embedding:0.6b", revision="test-revision",
                                             dimensions=1024, query_instruction="Find evidence."),
                     chunk_size_chars=18, overlap_chars=2)
    vectors = [[1.0] + [0.0] * 1023 for _ in prepared.chunks]
    return info, prepared, vectors


class NoDatabase:
    def acquire(self):
        pytest.fail("Invalid input must fail before accessing the database")


@pytest.mark.parametrize("change", ["missing-vector", "wrong-dimension", "zero", "nan", "bool",
                                    "wrong-chunk-text", "wrong-raw-text", "bad-offset", "duplicate-chunk",
                                    "wrong-chunk-order", "missing-page", "empty-chunks", "oversized-chunk",
                                    "forged-normalization", "wrong-overlap", "unknown-cleaner", "unknown-splitter"])
def test_invalid_import_is_rejected_before_database_access(change):
    from app.rag.store import PostgresPaperStore

    info, prepared, vectors = sample()
    if change == "missing-vector":
        vectors.pop()
    elif change == "wrong-dimension":
        vectors[0].pop()
    elif change == "zero":
        vectors[0] = [0.0] * 1024
    elif change == "nan":
        vectors[0][0] = float("nan")
    elif change == "bool":
        vectors[0][0] = True
    elif change == "wrong-chunk-text":
        prepared.chunks[0].text = "forged"
    elif change == "wrong-raw-text":
        prepared.chunks[0].raw_text = "forged"
    elif change == "bad-offset":
        prepared.chunks[0].raw_end_char = 999
    elif change == "duplicate-chunk":
        prepared.chunks.append(prepared.chunks[0].model_copy())
        vectors.append(vectors[0].copy())
    elif change == "wrong-chunk-order":
        prepared.chunks[0].chunk_index = 3
    elif change == "missing-page":
        prepared.pages = []
    elif change == "empty-chunks":
        prepared.chunks, vectors = [], []
    elif change == "oversized-chunk":
        info = info.model_copy(update={"chunk_size_chars": 3})
    elif change == "wrong-overlap":
        info = info.model_copy(update={"overlap_chars": 0})
    elif change == "unknown-cleaner":
        info = info.model_copy(update={"cleaner_version": "unverified-version"})
    elif change == "unknown-splitter":
        info = info.model_copy(update={"splitter_version": "unverified-version"})
    elif change == "forged-normalization":
        from app.rag.models import CleanedPage, PreparedText, TextChunk
        prepared = PreparedText(
            pages=[CleanedPage(page_number=1, raw_text="CAT", text="DOG", source_offsets=(0, 1, 2))],
            chunks=[TextChunk(page_number=1, chunk_index=0, text="DOG", raw_text="CAT",
                              start_char=0, end_char=3, raw_start_char=0, raw_end_char=3)])
        vectors = [vectors[0]]
    with pytest.raises(ValueError):
        asyncio.run(PostgresPaperStore(NoDatabase()).save(info, prepared, vectors))


@pytest.mark.parametrize("query", [[], [0.0] * 1024, [1.0, 2.0], [float("inf")] * 1024])
def test_invalid_query_vector_is_rejected_before_database_access(query):
    from app.rag.store import PostgresPaperStore

    info, _, _ = sample()
    with pytest.raises(ValueError):
        asyncio.run(PostgresPaperStore(NoDatabase()).search("paper", query, info.embedding))


@pytest.mark.parametrize("top_k", [0, -1, 101, True, 1.5])
def test_invalid_top_k_is_rejected_before_database_access(top_k):
    from app.rag.store import PostgresPaperStore

    info, _, vectors = sample()
    with pytest.raises(ValueError):
        asyncio.run(PostgresPaperStore(NoDatabase()).search("paper", vectors[0], info.embedding, top_k=top_k))


def test_schema_dimension_is_explicit_and_profile_is_immutable():
    from app.rag.storage_models import EmbeddingSpec

    with pytest.raises(ValueError):
        EmbeddingSpec(model="model", revision="rev", dimensions=3, query_instruction="Find evidence.")
    profile = sample()[0].embedding
    with pytest.raises(ValueError):
        profile.model = "changed"
