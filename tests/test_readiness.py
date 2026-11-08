"""Probe behavior, cancellation, and startup state without external services."""
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.testclient import TestClient
import httpx
import pytest
from pydantic import ValidationError

from app.api.routes.health import router
from app.config import Settings, get_settings


class ProbePool:
    def __init__(self, *, phase=None, error=None, value=1):
        self.phase = phase
        self.error = error
        self.value = value
        self.entered = asyncio.Event()
        self.acquires = 0
        self.releases = 0
        self.queries = []
        self.timeouts = []

    @asynccontextmanager
    async def acquire(self, *, timeout):
        self.acquires += 1
        self.timeouts.append(timeout)
        if self.phase == 'acquire':
            self.entered.set()
            await asyncio.Event().wait()
        try:
            yield self
        finally:
            self.releases += 1

    async def fetchval(self, query, *, timeout):
        self.queries.append(query)
        self.timeouts.append(timeout)
        if self.phase == 'query':
            self.entered.set()
            await asyncio.Event().wait()
        if self.error:
            raise self.error
        return self.value


def probe_app(*, pool=None, started=True, database=True, rag=False, service=False, timeout=0.03):
    app = FastAPI()
    app.include_router(router)
    app.state.startup_complete = started
    if pool is not None:
        app.state.db_pool = pool
    if service:
        app.state.rag_service = object()
    # Dependency override avoids the user's .env and global application state.
    settings = Settings(_env_file=None, database_url='postgresql://unused/test' if database else None,
                        rag_enabled=rag, readiness_timeout_seconds=timeout)
    app.dependency_overrides[get_settings] = lambda: settings
    return app


def test_ready_without_database_preserves_memory_mode():
    with TestClient(probe_app(database=False)) as client:
        response = client.get('/ready')
    assert response.status_code == 200
    assert response.json() == {'status': 'ready', 'checks': {
        'startup': 'complete', 'database': 'not_configured', 'rag': 'disabled'}}
    assert response.headers['cache-control'] == 'no-store'


def test_ready_queries_configured_pool_and_returns_connection():
    pool = ProbePool()
    with TestClient(probe_app(pool=pool, timeout=0.2)) as client:
        response = client.get('/ready')
    assert response.status_code == 200
    assert response.json()['checks']['database'] == 'ok'
    assert pool.queries == ['SELECT 1']
    assert pool.acquires == pool.releases == 1
    assert pool.timeouts == [0.2, 0.2]


def test_ready_fails_when_configured_pool_is_missing():
    response = TestClient(probe_app()).get('/ready')
    assert response.status_code == 503
    assert response.json()['checks']['database'] == 'unavailable'
    assert response.headers['cache-control'] == 'no-store'


def test_liveness_remains_ok_when_database_probe_fails():
    pool = ProbePool(error=RuntimeError('database down'))
    with TestClient(probe_app(pool=pool)) as client:
        assert client.get('/ready').status_code == 503
        response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}
    assert pool.acquires == pool.releases == 1


def test_unfinished_startup_skips_dependency_checks():
    pool = ProbePool()
    response = TestClient(probe_app(pool=pool, started=False)).get('/ready')
    assert response.status_code == 503
    assert response.json() == {'status': 'not_ready', 'checks': {
        'startup': 'incomplete', 'database': 'not_checked', 'rag': 'not_checked'}}
    assert pool.acquires == 0


@pytest.mark.parametrize('service,expected', [(True, 200), (False, 503)])
def test_rag_initialization_is_checked_only_when_enabled(service, expected):
    response = TestClient(probe_app(pool=ProbePool(), rag=True, service=service)).get('/ready')
    assert response.status_code == expected
    assert response.json()['checks']['rag'] == ('initialized' if service else 'unavailable')


@pytest.mark.parametrize('error', [RuntimeError, ConnectionError, TimeoutError])
def test_probe_failure_is_sanitized_and_releases_connection(error):
    secret = 'postgresql://user:do-not-expose@private-host/database'
    pool = ProbePool(error=error(secret))
    response = TestClient(probe_app(pool=pool)).get('/ready')
    assert response.status_code == 503
    assert response.json()['status'] == 'not_ready'
    assert secret not in response.text and 'private-host' not in response.text
    assert pool.releases == 1


def test_wrong_select_result_is_not_reported_as_ready():
    response = TestClient(probe_app(pool=ProbePool(value=None))).get('/ready')
    assert response.status_code == 503


def test_connection_return_failure_is_not_reported_as_ready_and_can_recover():
    class ReleaseFailingPool(ProbePool):
        fail_release = True

        @asynccontextmanager
        async def acquire(self, *, timeout):
            async with super().acquire(timeout=timeout) as connection:
                yield connection
            if self.fail_release:
                raise RuntimeError('private connection reset detail')

    pool = ReleaseFailingPool()
    with TestClient(probe_app(pool=pool)) as client:
        failed = client.get('/ready')
        assert failed.status_code == 503
        assert failed.json()['checks']['database'] == 'unavailable'
        assert 'private connection reset detail' not in failed.text
        pool.fail_release = False
        assert client.get('/ready').status_code == 200
    assert pool.queries == ['SELECT 1', 'SELECT 1']
    assert pool.acquires == pool.releases == 2


@pytest.mark.parametrize('phase', ['acquire', 'query'])
def test_probe_deadline_covers_pool_wait_and_query(phase):
    async def run():
        pool = ProbePool(phase=phase)
        app = probe_app(pool=pool)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            response = await asyncio.wait_for(client.get('/ready'), timeout=1.0)
        assert response.status_code == 503
        assert pool.entered.is_set()
        assert pool.releases == (1 if phase == 'query' else 0)
    asyncio.run(run())


@pytest.mark.parametrize('phase', ['acquire', 'query'])
def test_request_cancellation_propagates_without_leaking_a_connection(phase):
    async def run():
        pool = ProbePool(phase=phase)
        app = probe_app(pool=pool, timeout=1)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            task = asyncio.create_task(client.get('/ready'))
            try:
                await asyncio.wait_for(pool.entered.wait(), timeout=1)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
        assert pool.releases == (1 if phase == 'query' else 0)
    asyncio.run(run())


@pytest.mark.parametrize('timeout', [0, -1, 11, float('nan'), float('inf')])
def test_readiness_timeout_rejects_invalid_budgets(timeout):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, readiness_timeout_seconds=timeout)


def test_ready_openapi_documents_unavailable_response():
    schema = TestClient(probe_app()).get('/openapi.json').json()
    assert '503' in schema['paths']['/ready']['get']['responses']


def test_main_lifespan_sets_ready_only_after_startup_and_clears_before_pool_close(monkeypatch):
    from app import main
    settings = Settings(_env_file=None, database_url='postgresql://unused/test')
    instance = FastAPI()
    events = []

    class Pool:
        async def close(self):
            events.append(('close', instance.state.startup_complete))

    async def create_pool(**kwargs):
        events.append(('create', instance.state.startup_complete))
        return Pool()

    monkeypatch.setattr(main, 'get_settings', lambda: settings)
    monkeypatch.setattr(main.asyncpg, 'create_pool', create_pool)

    async def run():
        async with main.lifespan(instance):
            assert instance.state.startup_complete is True
        assert instance.state.startup_complete is False
    asyncio.run(run())
    assert events == [('create', False), ('close', False)]


def test_failed_startup_leaves_probe_not_ready(monkeypatch):
    from app import main
    settings = Settings(_env_file=None, database_url=None, rag_enabled=True)
    instance = FastAPI()
    monkeypatch.setattr(main, 'get_settings', lambda: settings)
    from app.rag.service_errors import RagUnavailableError

    async def run():
        with pytest.raises(RagUnavailableError):
            async with main.lifespan(instance):
                pytest.fail('Startup should not succeed')
        assert instance.state.startup_complete is False
    asyncio.run(run())
