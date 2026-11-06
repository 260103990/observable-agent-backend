import asyncio
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app


def configure(monkeypatch):
    from app import main
    events=[]
    service=object()
    class Pool:
        async def close(self):
            events.append(("pool.close",asyncio.get_running_loop()))
    class Http:
        async def __aenter__(self):
            events.append(("http.open",asyncio.get_running_loop()))
            return self
        async def __aexit__(self,*args):
            events.append(("http.close",asyncio.get_running_loop()))
    pool=Pool()
    async def create_pool(**kwargs):
        events.append(("pool.open",asyncio.get_running_loop()))
        return pool
    async def build(**kwargs):
        events.append(("service.build",asyncio.get_running_loop()))
        assert kwargs["pool"] is pool
        return service
    monkeypatch.setenv("DATABASE_URL","postgresql://agent:fake@db/test")
    monkeypatch.setenv("RAG_ENABLED","true")
    monkeypatch.setattr(main.asyncpg,"create_pool",create_pool)
    monkeypatch.setattr(main,"httpx",SimpleNamespace(AsyncClient=lambda **kwargs:Http(),Limits=lambda **kwargs:None),raising=False)
    monkeypatch.setattr(main,"build_rag_service",build,raising=False)
    get_settings.cache_clear()
    return main,events,service


def test_rag_lifespan_owns_pool_http_and_service_in_one_loop(monkeypatch):
    _,events,service=configure(monkeypatch)
    try:
        with TestClient(app):
            assert app.state.rag_service is service
            assert [n for n,_ in events]==["pool.open","http.open","service.build"]
        assert [n for n,_ in events]==["pool.open","http.open","service.build","http.close","pool.close"]
        assert len({id(loop) for _,loop in events})==1
        assert not hasattr(app.state,"rag_service")
        assert not hasattr(app.state,"db_pool")
    finally:
        get_settings.cache_clear()


def test_rag_initialization_failure_closes_all_resources(monkeypatch):
    from app.rag.service_errors import RagUnavailableError
    main,events,_=configure(monkeypatch)
    async def fail(**kwargs):
        raise RagUnavailableError("schema missing")
    monkeypatch.setattr(main,"build_rag_service",fail)
    try:
        with pytest.raises(RagUnavailableError):
            with TestClient(app):
                pytest.fail("Startup must fail")
        assert [n for n,_ in events]==["pool.open","http.open","http.close","pool.close"]
        assert not hasattr(app.state,"rag_service") and not hasattr(app.state,"db_pool")
    finally:
        get_settings.cache_clear()


def test_rag_requires_database_when_explicitly_enabled(monkeypatch):
    from app.rag.service_errors import RagUnavailableError
    monkeypatch.delenv("DATABASE_URL",raising=False)
    monkeypatch.setenv("RAG_ENABLED","true")
    get_settings.cache_clear()
    try:
        with pytest.raises(RagUnavailableError):
            with TestClient(app):
                pytest.fail("RAG cannot run with no persistent database")
    finally:
        get_settings.cache_clear()
