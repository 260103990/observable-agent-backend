"""Real PostgreSQL + HTTP integration; remote models remain controlled doubles."""
import asyncio
import json
import os
from pathlib import Path

import asyncpg
from fastapi.testclient import TestClient
import httpx
import pytest

from app.config import get_settings
from app.main import app
from tests.pdf_helpers import write_text_pdf


pytestmark=pytest.mark.postgres
ROOT=Path(__file__).parents[2]


def initialize_database():
    async def run():
        c=await asyncpg.connect(os.environ["TEST_PGVECTOR_DATABASE_URL"])
        try:
            await c.execute((ROOT/"app/rag/schema.sql").read_text(encoding="utf-8"))
            await c.execute((ROOT/"app/observability/schema.sql").read_text(encoding="utf-8"))
        finally:
            await c.close()
    asyncio.run(run())


def cleanup(papers,traces):
    async def run():
        c=await asyncpg.connect(os.environ["TEST_PGVECTOR_DATABASE_URL"])
        try:
            await c.execute("DELETE FROM rag_papers WHERE paper_id=ANY($1::text[])",papers)
            await c.execute("DELETE FROM traces WHERE trace_id=ANY($1::text[])",traces)
        finally:
            await c.close()
    asyncio.run(run())


def configure(monkeypatch,tmp_path,*,fail_embedding=False):
    from app import main
    original=httpx.AsyncClient
    calls=[]
    def handler(request):
        calls.append(request.url.path)
        if request.url.path=="/api/tags":
            return httpx.Response(200,json={"models":[{"name":"qwen3-embedding:0.6b","digest":"test-revision"}]})
        if request.url.path=="/api/embed":
            if fail_embedding:
                raise httpx.ReadTimeout("simulated",request=request)
            body=json.loads(request.content)
            return httpx.Response(200,json={"model":"qwen3-embedding:0.6b",
                "embeddings":[[1.0]+[0.0]*1023 for _ in body["input"]]})
        if request.url.path=="/chat/completions":
            return httpx.Response(200,json={"id":"test-answer","model":"deepseek-flash","choices":[
                {"finish_reason":"stop","message":{"role":"assistant","content":json.dumps({
                    "claims":[{"text":"Paper evidence supplied.","evidence_ids":["E1"]}],
                    "insufficient_evidence":False,"reason":None})}}]})
        raise AssertionError("Unexpected model endpoint")
    monkeypatch.setattr(main.httpx,"AsyncClient",lambda **kwargs:original(transport=httpx.MockTransport(handler),**kwargs))
    monkeypatch.setenv("DATABASE_URL",os.environ["TEST_PGVECTOR_DATABASE_URL"])
    monkeypatch.setenv("RAG_ENABLED","true")
    monkeypatch.setenv("RAG_STORAGE_DIR",str(tmp_path/"uploaded"))
    monkeypatch.setenv("DEEPSEEK_API_KEY","integration-fake-key")
    get_settings.cache_clear()
    return calls


def test_http_import_question_and_application_restart_use_real_persistence(monkeypatch,tmp_path):
    initialize_database()
    calls=configure(monkeypatch,tmp_path)
    fixture=tmp_path/"paper.pdf"
    write_text_pdf(fixture,["Paper cover",None,"Baseline throughput: 42 requests/s."])
    papers,traces=[],[]
    try:
        with TestClient(app) as client:
            imported=client.post("/papers",files={"file":("paper.pdf",fixture.read_bytes(),"application/pdf")},
                                 data={"title":"HTTP fixture"},headers={"X-Request-ID":"import-request"})
            assert imported.status_code==201,imported.text
            info=imported.json()
            papers.append(info["paper_id"])
            traces.append(info["trace_id"])
            assert imported.headers["x-trace-id"]==info["trace_id"]
            assert (info["page_count"],info["chunk_count"])==(3,2)
            asked=client.post(f"/papers/{info['paper_id']}/questions",json={"question":"What is the baseline?","top_k":2})
            assert asked.status_code==200,asked.text
            traces.append(asked.json()["trace_id"])
            trace=client.get("/traces/"+info["trace_id"]).json()
            assert trace["status"]=="ok" and trace["request_id"]=="import-request"
            assert trace["spans"][0]["name"]=="rag.import" and len(trace["spans"])==7
        # A new lifespan creates an independent pool and HTTP client, like an app restart.
        with TestClient(app) as restarted:
            assert restarted.get("/traces/"+info["trace_id"]).status_code==200
            asked=restarted.post(f"/papers/{info['paper_id']}/questions",json={"question":"What is the baseline?"})
            assert asked.status_code==200,asked.text
            traces.append(asked.json()["trace_id"])
            assert asked.json()["citations"][0]["paper_id"]==info["paper_id"]
            assert restarted.get("/traces/"+asked.json()["trace_id"]).json()["spans"][0]["name"]=="rag.ask"
        # One document batch and two question requests; no document re-embedding after restart.
        assert calls.count("/api/embed")==3
    finally:
        get_settings.cache_clear()
        cleanup(papers,traces)


def test_real_database_failure_trace_and_upload_cleanup_on_provider_timeout(monkeypatch,tmp_path):
    initialize_database()
    configure(monkeypatch,tmp_path,fail_embedding=True)
    fixture=tmp_path/"paper.pdf"
    write_text_pdf(fixture,["Text"])
    traces=[]
    try:
        with TestClient(app) as client:
            response=client.post("/papers",files={"file":("paper.pdf",fixture.read_bytes(),"application/pdf")})
            assert response.status_code==504,response.text
            trace_id=response.headers["x-trace-id"]
            traces.append(trace_id)
            record=client.get("/traces/"+trace_id).json()
            assert record["status"]=="error"
            assert record["spans"][-1]["name"]=="rag.documents.embed"
            assert record["spans"][-1]["error_type"]=="EmbeddingTimeoutError"
        assert not list((tmp_path/"uploaded").glob("*.pdf"))
    finally:
        get_settings.cache_clear()
        cleanup([],traces)


def test_metadata_only_store_read_uses_real_database():
    from app.rag.store import PostgresPaperStore
    from tests.test_paper_store import sample
    from uuid import uuid4
    initialize_database()
    info,prepared,vectors=sample("metadata-"+uuid4().hex)
    async def run():
        async with asyncpg.create_pool(os.environ["TEST_PGVECTOR_DATABASE_URL"],min_size=1,max_size=1) as pool:
            store=PostgresPaperStore(pool)
            try:
                assert await store.get_info("missing-"+uuid4().hex) is None
                await store.save(info,prepared,vectors)
                assert await store.get_info(info.paper_id)==info
            finally:
                await pool.execute("DELETE FROM rag_papers WHERE paper_id=$1",info.paper_id)
    asyncio.run(run())
