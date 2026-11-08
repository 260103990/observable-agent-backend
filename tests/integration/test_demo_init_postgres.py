"""Real bootstrap SQL in newly owned disposable databases; never existing schemas."""
import asyncio
from contextlib import asynccontextmanager
import importlib.util
import os
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import asyncpg
import pytest


pytestmark = pytest.mark.postgres
ROOT = Path(__file__).parents[2]


@asynccontextmanager
async def owned_database(monkeypatch):
    admin_dsn = os.environ.get('TEST_PGVECTOR_DATABASE_URL')
    if not admin_dsn:
        pytest.skip('Set TEST_PGVECTOR_DATABASE_URL to a dedicated test database with CREATEDB rights')
    parsed = urlsplit(admin_dsn)
    if not parsed.path.rstrip('/').endswith('_test'):
        pytest.fail('Bootstrap integration requires a dedicated test admin database ending in _test')
    original_connect = asyncpg.connect
    admin = await original_connect(admin_dsn)
    name = 'oab_demo_init_' + uuid4().hex + '_test'
    created = False
    try:
        await admin.execute(f'CREATE DATABASE "{name}"')
        created = True
        target = urlunsplit(parsed._replace(path='/' + name))
        spec = importlib.util.spec_from_file_location('demo_init_integration', ROOT/'scripts/init_demo_database.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        async def redirect_demo_connection(dsn, **kwargs):
            # Only the test redirects the already validated Compose-only URL.
            assert urlsplit(dsn).hostname == 'postgres'
            return await original_connect(target, **kwargs)

        monkeypatch.setattr(mod.asyncpg, 'connect', redirect_demo_connection)
        inspect = await original_connect(target)
        try:
            yield mod, inspect
        finally:
            await inspect.close()
    finally:
        if created:
            assert name.startswith('oab_demo_init_') and name.endswith('_test') and len(name) == 51
            await admin.execute(f'DROP DATABASE "{name}"')
        await admin.close()


DSN = 'postgresql://agent:integration-only@postgres:5432/observable_agent_demo'


def test_real_demo_init_is_repeatable_without_replacing_trace(monkeypatch):
    async def run():
        async with owned_database(monkeypatch) as (mod, inspect):
            assert await mod.initialize_demo_database(DSN) == 'initialized'
            await inspect.execute("INSERT INTO traces VALUES ('retained', 'request', 'ok', 1)")
            assert await mod.initialize_demo_database(DSN) == 'already_initialized'
            assert await inspect.fetchval("SELECT count(*) FROM traces WHERE trace_id='retained'") == 1
            assert await inspect.fetchval("SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                                          "WHERE attrelid='rag_chunks'::regclass AND attname='embedding'") == 'vector(1024)'
    asyncio.run(run())


def test_real_concurrent_demo_initializers_are_serialized(monkeypatch):
    async def run():
        async with owned_database(monkeypatch) as (mod, inspect):
            outcomes = await asyncio.gather(mod.initialize_demo_database(DSN), mod.initialize_demo_database(DSN))
            assert sorted(outcomes) == ['already_initialized', 'initialized']
            assert await inspect.fetchval("SELECT count(*) FROM pg_tables WHERE schemaname='public'") == 5
    asyncio.run(run())


def test_real_failure_in_second_schema_rolls_back_first_schema(monkeypatch):
    async def run():
        async with owned_database(monkeypatch) as (mod, inspect):
            original_body = mod._schema_body
            def fail_second(path):
                return 'SELECT 1 / 0;' if path.parent.name == 'rag' else original_body(path)
            monkeypatch.setattr(mod, '_schema_body', fail_second)
            with pytest.raises(asyncpg.DivisionByZeroError):
                await mod.initialize_demo_database(DSN)
            assert await inspect.fetchval("SELECT count(*) FROM pg_tables WHERE schemaname='public'") == 0
            monkeypatch.setattr(mod, '_schema_body', original_body)
            assert await mod.initialize_demo_database(DSN) == 'initialized'
    asyncio.run(run())


def test_real_partial_schema_is_rejected_without_overwrite(monkeypatch):
    async def run():
        async with owned_database(monkeypatch) as (mod, inspect):
            await inspect.execute('CREATE TABLE traces (sentinel INTEGER)')
            await inspect.execute('INSERT INTO traces VALUES (7)')
            with pytest.raises(mod.DemoInitError, match='schema_requires_review'):
                await mod.initialize_demo_database(DSN)
            assert await inspect.fetchval('SELECT sentinel FROM traces') == 7
            assert await inspect.fetchval("SELECT count(*) FROM pg_tables WHERE schemaname='public'") == 1
    asyncio.run(run())


def test_real_demo_tables_use_public_even_if_connection_search_path_differs(monkeypatch):
    async def run():
        async with owned_database(monkeypatch) as (mod, inspect):
            await inspect.execute('CREATE SCHEMA shadow')
            redirected = mod.asyncpg.connect
            async def shadow_connect(dsn, **kwargs):
                return await redirected(dsn, server_settings={'search_path': 'shadow,public'}, **kwargs)
            monkeypatch.setattr(mod.asyncpg, 'connect', shadow_connect)
            assert await mod.initialize_demo_database(DSN) == 'initialized'
            assert await inspect.fetchval("SELECT count(*) FROM pg_tables WHERE schemaname='public'") == 5
            assert await inspect.fetchval("SELECT count(*) FROM pg_tables WHERE schemaname='shadow'") == 0
    asyncio.run(run())
