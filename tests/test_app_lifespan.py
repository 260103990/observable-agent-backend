import asyncio

import asyncpg
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.observability.store import PostgresTraceStore


def test_lifespan_creates_and_closes_postgres_pool(monkeypatch):
    database_url = (
        "postgresql://agent:test-password@db:5432/test_db"
    )
    events = []

    class FakePool:
        async def close(self):
            events.append(("close", asyncio.get_running_loop()))

    fake_pool = FakePool()

    async def fake_create_pool(dsn, **kwargs):
        assert dsn == database_url
        events.append(("create", asyncio.get_running_loop()))
        return fake_pool

    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setattr(
        asyncpg,
        "create_pool",
        fake_create_pool,
    )

    get_settings.cache_clear()

    try:
        with TestClient(app):
            assert [name for name, _ in events] == ["create"]
            assert app.state.db_pool is fake_pool
            assert isinstance(
                app.state.trace_store,
                PostgresTraceStore,
            )

        assert [name for name, _ in events] == [
            "create",
            "close",
        ]
        assert events[0][1] is events[1][1]
        assert not hasattr(app.state, "db_pool")
        assert not hasattr(app.state, "trace_store")
    finally:
        get_settings.cache_clear()