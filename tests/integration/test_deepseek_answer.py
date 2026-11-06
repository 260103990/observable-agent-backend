"""Explicit opt-in paid API tests. Never run a real request by default."""
import asyncio
import os

import httpx
import pytest

from tests.answer_helpers import hit


pytestmark = pytest.mark.skipif(os.environ.get("RUN_DEEPSEEK_TESTS") != "1",
                              reason="Set RUN_DEEPSEEK_TESTS=1 for real DeepSeek API validation")


@pytest.mark.parametrize("question,supported", [("基线吞吐量是多少？", True), ("实验使用的GPU具体型号是什么？", False)])
def test_real_deepseek_supported_and_missing_evidence(question, supported):
    from app.config import Settings
    from app.rag.answer import DeepSeekAnswerClient

    settings = Settings()
    if settings.deepseek_api_key is None:
        pytest.skip("Local DEEPSEEK_API_KEY is not configured")
    async def run():
        async with httpx.AsyncClient(trust_env=False) as http:
            result = await DeepSeekAnswerClient.from_settings(settings, client=http).answer("paper-1", question, [hit()])
            if supported:
                assert not result.insufficient_evidence
                assert "42" in result.answer and len(result.citations) == 1
                assert result.citations[0].raw_text == hit().chunk.raw_text
            else:
                assert result.insufficient_evidence
            print(result.answer)
    asyncio.run(run())


@pytest.mark.skipif(os.environ.get("RUN_DEEPSEEK_PAPER_TESTS") != "1",
                    reason="Set RUN_DEEPSEEK_PAPER_TESTS=1 with RAG_DEVELOPMENT_DATABASE_URL and PAPER_ID")
def test_real_persisted_paper_retrieval_to_deepseek_answer():
    import asyncpg
    import json
    from app.config import Settings
    from app.rag.answer import DeepSeekAnswerClient
    from app.rag.embedding import OllamaEmbeddingClient
    from app.rag.store import PostgresPaperStore

    settings = Settings()
    if settings.deepseek_api_key is None:
        pytest.skip("Local DEEPSEEK_API_KEY is not configured")

    async def run():
        async with asyncpg.create_pool(os.environ["RAG_DEVELOPMENT_DATABASE_URL"], min_size=1, max_size=2) as pool:
            store = PostgresPaperStore(pool)
            paper = await store.get(os.environ["PAPER_ID"])
            assert paper is not None
            question = "Cortex在实验中使用了哪些比较基线？请保留基线名称及适用条件。"
            async with httpx.AsyncClient(trust_env=False) as http:
                embedding = OllamaEmbeddingClient.from_settings(settings, client=http)
                assert embedding.model == paper.paper.embedding.model
                assert embedding.query_instruction == paper.paper.embedding.query_instruction
                tags = await http.get("http://127.0.0.1:11434/api/tags", timeout=10)
                tags.raise_for_status()
                assert any(m["name"] == embedding.model and m["digest"] == paper.paper.embedding.revision
                           for m in tags.json()["models"])
                query = await embedding.embed_query(question)
                hits = await store.search(paper.paper.paper_id, query, paper.paper.embedding)
                result = await DeepSeekAnswerClient.from_settings(settings, client=http).answer(paper.paper.paper_id, question, hits)
            sources = {(h.chunk.page_number, h.chunk.chunk_index): h.chunk for h in hits}
            for citation in result.citations:
                assert citation.raw_text == sources[(citation.page_number, citation.chunk_index)].raw_text
            print(json.dumps({"answer": result.answer, "insufficient_evidence": result.insufficient_evidence,
                "citations": [{"id": c.evidence_id, "page": c.page_number, "index": c.chunk_index,
                               "raw_text": c.raw_text} for c in result.citations]}, ensure_ascii=True))
    asyncio.run(run())
