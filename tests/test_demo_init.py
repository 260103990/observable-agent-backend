"""Explicit demo bootstrap: target isolation, transaction boundaries, no hidden migration."""
import asyncio
from contextlib import asynccontextmanager
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]


def module():
    path = ROOT/'scripts/init_demo_database.py'
    assert path.is_file(), 'Explicit demo initializer is not implemented'
    spec = importlib.util.spec_from_file_location('demo_init_under_test', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class Connection:
    def __init__(self, existing=(), fail_ddl=False):
        self.existing = existing
        self.fail_ddl = fail_ddl
        self.events = []

    @asynccontextmanager
    async def transaction(self):
        self.events.append('begin')
        try:
            yield
        except BaseException:
            self.events.append('rollback')
            raise
        else:
            self.events.append('commit')

    async def execute(self, query, *args):
        if query == 'SET LOCAL search_path = public':
            self.events.append('public_schema')
        elif 'pg_advisory_xact_lock' in query:
            self.events.append('lock')
        else:
            self.events.append(('ddl', query))
            if self.fail_ddl:
                raise RuntimeError('private database detail')

    async def fetch(self, query):
        self.events.append('inspect')
        return [{'tablename': name} for name in self.existing]

    async def close(self, *, timeout):
        self.events.append('close')


def connect_fake(monkeypatch, mod, connection):
    async def connect(*args, **kwargs):
        return connection
    monkeypatch.setattr(mod.asyncpg, 'connect', connect)


@pytest.mark.parametrize('dsn', [None, '', 'postgresql://agent:x@127.0.0.1:55432/observable_agent',
    'postgresql://agent:x@postgres/observable_agent', 'postgresql://agent:x@other/observable_agent_demo',
    'postgresql://agent:x@postgres:55433/observable_agent_demo',
    'postgresql://agent:x@postgres/observable_agent_demo?host=production',
    'postgresql://agent:x@postgres/observable_agent_demo#other', 'https://postgres/observable_agent_demo'])
def test_unsafe_targets_are_rejected_before_connect(monkeypatch, dsn):
    mod = module()
    async def forbidden(*args, **kwargs):
        pytest.fail('Unsafe target must not open a connection')
    monkeypatch.setattr(mod.asyncpg, 'connect', forbidden)
    with pytest.raises(mod.DemoInitError, match='invalid_target'):
        asyncio.run(mod.initialize_demo_database(dsn))


def test_empty_demo_database_initializes_both_schemas_in_one_transaction(monkeypatch):
    mod = module()
    connection = Connection()
    connect_fake(monkeypatch, mod, connection)
    assert asyncio.run(mod.initialize_demo_database('postgresql://agent:x@postgres:5432/observable_agent_demo')) == 'initialized'
    assert connection.events[:4] == ['begin', 'public_schema', 'lock', 'inspect']
    ddl = [event[1] for event in connection.events if isinstance(event, tuple)]
    assert len(ddl) == 2
    assert 'CREATE TABLE IF NOT EXISTS traces' in ddl[0]
    assert 'CREATE TABLE IF NOT EXISTS rag_papers' in ddl[1]
    assert all('BEGIN;' not in sql and 'COMMIT;' not in sql for sql in ddl)
    assert connection.events[-2:] == ['commit', 'close']


def test_repeated_bootstrap_does_not_execute_ddl(monkeypatch):
    mod = module()
    connection = Connection(['traces', 'spans', 'rag_papers', 'rag_pages', 'rag_chunks'])
    connect_fake(monkeypatch, mod, connection)
    assert asyncio.run(mod.initialize_demo_database('postgresql://agent:x@postgres/observable_agent_demo')) == 'already_initialized'
    assert connection.events == ['begin', 'public_schema', 'lock', 'inspect', 'commit', 'close']


@pytest.mark.parametrize('existing', [['traces'], ['unrelated_table'], ['traces', 'spans', 'rag_papers', 'rag_pages']])
def test_partial_or_unexpected_schema_requires_explicit_review(monkeypatch, existing):
    mod = module()
    connection = Connection(existing)
    connect_fake(monkeypatch, mod, connection)
    with pytest.raises(mod.DemoInitError, match='schema_requires_review'):
        asyncio.run(mod.initialize_demo_database('postgresql://agent:x@postgres/observable_agent_demo'))
    assert connection.events == ['begin', 'public_schema', 'lock', 'inspect', 'rollback', 'close']


def test_ddl_failure_rolls_back_and_closes_connection(monkeypatch):
    mod = module()
    connection = Connection(fail_ddl=True)
    connect_fake(monkeypatch, mod, connection)
    with pytest.raises(RuntimeError):
        asyncio.run(mod.initialize_demo_database('postgresql://agent:x@postgres/observable_agent_demo'))
    assert connection.events[-2:] == ['rollback', 'close']
    assert 'commit' not in connection.events


def test_unconfirmed_cli_never_connects(monkeypatch, capsys):
    mod = module()
    monkeypatch.setattr('sys.argv', ['init_demo_database.py'])
    with pytest.raises(SystemExit) as exit_info:
        mod.main()
    assert exit_info.value.code == 2


def test_cli_failure_does_not_print_database_credentials(monkeypatch, capsys):
    mod = module()
    monkeypatch.setattr('sys.argv', ['init_demo_database.py', '--confirm-demo'])
    monkeypatch.setenv('DATABASE_URL', 'postgresql://agent:private-password@postgres/observable_agent_demo')
    async def fail(dsn):
        raise RuntimeError(dsn)
    monkeypatch.setattr(mod, 'initialize_demo_database', fail)
    assert mod.main() == 1
    output = capsys.readouterr()
    assert 'private-password' not in output.err + output.out
    assert 'RuntimeError' in output.err
