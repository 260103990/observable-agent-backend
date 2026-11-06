import asyncio
import json

import httpx
import pytest


def test_embedding_revision_uses_configured_origin_and_model():
    from app.rag.embedding import OllamaEmbeddingClient
    requests=[]
    def handler(request):
        requests.append(request)
        return httpx.Response(200,json={"models":[{"name":"other","digest":"other"},
                                                   {"name":"qwen3-embedding:0.6b","digest":"revision-1"}]})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            embedding=OllamaEmbeddingClient(client=http,embed_url="http://configured:11434/api/embed",timeout_seconds=4)
            assert await embedding.get_revision()=="revision-1"
            assert not http.is_closed
    asyncio.run(run())
    assert str(requests[0].url)=="http://configured:11434/api/tags"
    assert requests[0].extensions["timeout"]["read"]==4


@pytest.mark.parametrize("payload", [None,{}, {"models":[]}, {"models":[{"name":"qwen3-embedding:0.6b","digest":" "}]}])
def test_embedding_revision_rejects_missing_identity(payload):
    from app.rag.embedding import EmbeddingResponseError, OllamaEmbeddingClient
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r:httpx.Response(200,json=payload))) as http:
            with pytest.raises(EmbeddingResponseError):
                await OllamaEmbeddingClient(client=http).get_revision()
    asyncio.run(run())


def test_rag_settings_are_opt_in_and_validate_limits():
    from app.config import Settings
    settings=Settings(_env_file=None)
    assert settings.rag_enabled is False
    assert settings.rag_max_file_bytes==20*1024*1024
    assert settings.rag_max_pages==100
    assert settings.rag_max_concurrency==2
    for kwargs in ({"rag_max_pages":0},{"rag_queue_timeout_seconds":0},{"rag_max_chunks":0},
                   {"rag_chunk_size_chars":100,"rag_overlap_chars":100}):
        with pytest.raises(ValueError):
            Settings(_env_file=None,**kwargs)


def test_store_get_info_reads_only_the_metadata_row():
    from app.rag.store import PostgresPaperStore
    class Connection:
        async def fetchrow(self,sql,paper_id):
            assert "rag_papers" in sql and paper_id=="missing"
            return None
    class Acquire:
        async def __aenter__(self):
            return Connection()
        async def __aexit__(self,*args):
            return False
    class Pool:
        def acquire(self):
            return Acquire()
    assert asyncio.run(PostgresPaperStore(Pool()).get_info("missing")) is None
