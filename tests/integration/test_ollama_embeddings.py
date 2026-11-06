"""Explicit opt-in; never download models or call a service in default tests."""
import asyncio
import math
import os

import httpx
import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_OLLAMA_EMBEDDING_TESTS") != "1",
    reason="Set RUN_OLLAMA_EMBEDDING_TESTS=1 to test an installed local Ollama model",
)


def test_real_ollama_embeddings_have_expected_dimension_and_cross_language_ranking():
    from app.config import Settings
    from app.rag.embedding import OllamaEmbeddingClient

    documents = [
        "Cache hits let the agent reuse stored web search results without sending another external search request.",
        "The experiment studies PostgreSQL database connection pooling and transaction isolation levels.",
        "Vacuum equipment measures X-ray photoelectron spectroscopy spectra.",
    ]

    async def run():
        settings = Settings(_env_file=None)
        async with httpx.AsyncClient(trust_env=False) as http:
            embedding = OllamaEmbeddingClient.from_settings(settings, client=http)
            vectors = await embedding.embed_documents(documents)
            for vector in vectors:
                assert len(vector) == settings.ollama_embedding_dimensions
                assert math.isclose(math.hypot(*vector), 1.0, abs_tol=0.01)
            for query, expected in [
                ("命中搜索缓存后，还需要再次发送外部网络搜索请求吗？", 0),
                ("Which experiment studies database connection pooling?", 1),
            ]:
                vector = await embedding.embed_query(query)
                scores = [sum(x * y for x, y in zip(vector, doc)) /
                          (math.hypot(*vector) * math.hypot(*doc)) for doc in vectors]
                assert max(range(len(scores)), key=scores.__getitem__) == expected, scores
    asyncio.run(run())


@pytest.mark.skipif(not os.environ.get("PAPER_PDF_PATH"), reason="Set PAPER_PDF_PATH for a real paper integrity check")
def test_real_paper_chunks_can_be_embedded_and_scored():
    """Real model/text integration; not a recall or answer-quality benchmark."""
    import json
    from pathlib import Path
    from app.config import Settings
    from app.rag.embedding import OllamaEmbeddingClient
    from app.rag.pdf import extract_pdf_pages
    from app.rag.preparation import prepare_pages

    prepared = prepare_pages(extract_pdf_pages(Path(os.environ["PAPER_PDF_PATH"])))

    async def run():
        settings = Settings(_env_file=None)
        async with httpx.AsyncClient(trust_env=False) as http:
            embedding = OllamaEmbeddingClient.from_settings(settings, client=http)
            vectors = await embedding.embed_documents([chunk.text for chunk in prepared.chunks])
            assert len(vectors) == len(prepared.chunks)
            query = await embedding.embed_query("论文的实验使用了什么模型、数据集和比较基线？")
            scores = [sum(x * y for x, y in zip(query, vector)) /
                      (math.hypot(*query) * math.hypot(*vector)) for vector in vectors]
            assert all(math.isfinite(score) for score in scores)
            top = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:5]
            print(json.dumps({
                "pages": len(prepared.pages), "chunks": len(vectors), "dimensions": len(query),
                "query": "论文的实验使用了什么模型、数据集和比较基线？",
                "top5": [{"page_number": prepared.chunks[i].page_number,
                          "chunk_index": prepared.chunks[i].chunk_index,
                          "cosine": round(scores[i], 6)} for i in top],
                "scope": "In-memory integration diagnostic; no pgvector or answer validation",
            }, ensure_ascii=True))
    asyncio.run(run())
