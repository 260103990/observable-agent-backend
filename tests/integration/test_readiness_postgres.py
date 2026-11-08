"""Real database faults on an owned one-connection pool; no schema/data writes."""
import asyncio
import os
from urllib.parse import urlsplit

import asyncpg
import httpx
import pytest

from app.config import get_settings
from app.main import app


pytestmark = pytest.mark.postgres


@pytest.fixture
def database_probe(monkeypatch):
    from app import main
    dsn = os.environ.get('TEST_PGVECTOR_DATABASE_URL')
    if not dsn:
        pytest.skip('Set TEST_PGVECTOR_DATABASE_URL to a dedicated test database')
    if not urlsplit(dsn).path.rstrip('/').endswith('_test'):
        pytest.fail('Readiness fault tests require a database name ending in _test')
    monkeypatch.setenv('DATABASE_URL', dsn)
    monkeypatch.setenv('RAG_ENABLED', 'false')
    monkeypatch.setenv('WEB_SEARCH_CACHE_ENABLED', 'false')
    monkeypatch.setenv('READINESS_TIMEOUT_SECONDS', '0.15')
    original_create_pool = asyncpg.create_pool
    control = {'delay': False, 'disconnect': False, 'started': None}

    class FaultConnection(asyncpg.Connection):
        async def fetchval(self, query, *args, **kwargs):
            if query == 'SELECT 1':
                if control['disconnect']:
                    self.terminate()  # Only this test's own connection.
                elif control['delay']:
                    control['started'].set()
                    # Real PostgreSQL work and driver cancellation, rather
                    # than substituting an asyncio sleep for network I/O.
                    await super().fetchval('SELECT pg_sleep(5)', **kwargs)
            return await super().fetchval(query, *args, **kwargs)

    async def create_pool(**kwargs):
        return await original_create_pool(**kwargs, min_size=1, max_size=1, connection_class=FaultConnection)

    monkeypatch.setattr(main.asyncpg, 'create_pool', create_pool)
    get_settings.cache_clear()
    try:
        yield control
    finally:
        get_settings.cache_clear()


def test_real_pool_exhaustion_fails_readiness_then_recovers(database_probe):
    async def run():
        async with app.router.lifespan_context(app):
            pool = app.state.db_pool
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                assert (await client.get('/ready')).status_code == 200
                async with pool.acquire(timeout=1):
                    unavailable = await asyncio.wait_for(client.get('/ready'), timeout=2)
                    assert unavailable.status_code == 503
                    assert unavailable.json()['checks']['database'] == 'unavailable'
                    assert (await client.get('/health')).status_code == 200
                assert (await client.get('/ready')).status_code == 200
                assert pool.get_idle_size() == pool.get_size() == 1
    asyncio.run(run())


def test_real_slow_query_is_cancelled_and_pool_recovers(database_probe):
    async def run():
        database_probe['started'] = asyncio.Event()
        async with app.router.lifespan_context(app):
            pool = app.state.db_pool
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                database_probe['delay'] = True
                response = await asyncio.wait_for(client.get('/ready'), timeout=2)
                assert response.status_code == 503
                assert database_probe['started'].is_set()
                database_probe['delay'] = False
                assert (await client.get('/ready')).status_code == 200
                assert pool.get_idle_size() == pool.get_size() == 1
    asyncio.run(run())


def test_real_query_request_cancellation_returns_connection(database_probe):
    async def run():
        database_probe['started'] = asyncio.Event()
        async with app.router.lifespan_context(app):
            pool = app.state.db_pool
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                database_probe['delay'] = True
                task = asyncio.create_task(client.get('/ready'))
                try:
                    await asyncio.wait_for(database_probe['started'].wait(), timeout=1)
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await asyncio.wait_for(task, timeout=2)
                finally:
                    database_probe['delay'] = False
                    if not task.done():
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                assert (await client.get('/ready')).status_code == 200
                assert pool.get_idle_size() == pool.get_size() == 1
    asyncio.run(run())


def test_own_connection_disconnect_fails_probe_then_reconnects(database_probe):
    async def run():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                database_probe['disconnect'] = True
                assert (await client.get('/ready')).status_code == 503
                database_probe['disconnect'] = False
                assert (await client.get('/ready')).status_code == 200
                assert app.state.db_pool.get_idle_size() == 1
    asyncio.run(run())
