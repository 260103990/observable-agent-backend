"""Explicit initializer for the isolated Compose demo, never a migration tool."""
import argparse
import asyncio
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit

import asyncpg


ROOT = Path(__file__).resolve().parents[1]
TABLES = frozenset(('traces', 'spans', 'rag_papers', 'rag_pages', 'rag_chunks'))
DEMO_SCHEMA_LOCK_KEY = 0x4F414244454D4F  # OABDEMO, scoped to the target database.


class DemoInitError(Exception):
    """Contains only a safe, fixed error code."""


def _validate_target(dsn):
    try:
        parsed = urlsplit(dsn) if isinstance(dsn, str) else None
        if (parsed is None or parsed.scheme not in ('postgresql', 'postgres')
                or parsed.hostname != 'postgres' or parsed.port not in (None, 5432)
                or parsed.path != '/observable_agent_demo' or parsed.query or parsed.fragment):
            raise ValueError
    except ValueError:
        raise DemoInitError('invalid_target') from None


def _schema_body(path):
    lines = path.read_text(encoding='utf-8').splitlines()
    statements = [i for i, line in enumerate(lines) if line.strip() and not line.lstrip().startswith('--')]
    if not statements or lines[statements[0]].strip() != 'BEGIN;' or lines[statements[-1]].strip() != 'COMMIT;':
        raise DemoInitError('unexpected_schema_wrapper')
    return '\n'.join(lines[statements[0]+1:statements[-1]])


async def initialize_demo_database(dsn):
    _validate_target(dsn)
    scripts = [_schema_body(ROOT/name) for name in ('app/observability/schema.sql', 'app/rag/schema.sql')]
    connection = await asyncpg.connect(dsn, timeout=10, command_timeout=30)
    try:
        async with asyncio.timeout(60):
            async with connection.transaction():
                # The app checks public tables explicitly; do not inherit a
                # role/session search_path that creates them elsewhere.
                await connection.execute('SET LOCAL search_path = public')
                await connection.execute('SELECT pg_advisory_xact_lock($1)', DEMO_SCHEMA_LOCK_KEY)
                existing = {row['tablename'] for row in await connection.fetch(
                    "SELECT tablename FROM pg_tables WHERE schemaname='public'")}
                if existing == TABLES:
                    return 'already_initialized'
                if existing:
                    raise DemoInitError('schema_requires_review')
                for sql in scripts:
                    await connection.execute(sql)
                return 'initialized'
    finally:
        await connection.close(timeout=5)


def main():
    parser = argparse.ArgumentParser(description='Initialize the isolated demo database; not a migration command.')
    parser.add_argument('--confirm-demo', action='store_true', required=True)
    parser.parse_args()
    try:
        result = asyncio.run(initialize_demo_database(os.environ.get('DATABASE_URL')))
    except DemoInitError as exc:
        print('demo_init_failed: ' + str(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        # Database/driver errors may contain connection information.
        print('demo_init_failed: ' + type(exc).__name__, file=sys.stderr)
        return 1
    print('demo_init: ' + result)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
