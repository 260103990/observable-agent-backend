"""Real interleaving at the two SELECT boundary; no sleeps or existing data overwrite."""
import asyncio
from contextlib import asynccontextmanager
import os
from uuid import uuid4

import asyncpg
import pytest

from app.observability.models import SpanRecord,TraceRecord
from app.observability.store import PostgresTraceStore

pytestmark=pytest.mark.postgres


def trace(identifier,version):
    return TraceRecord(trace_id=identifier,request_id=version,status='ok',duration_ms=1,
        spans=[SpanRecord(span_id=version,parent_span_id=None,name=version,status='ok',duration_ms=1,
                          attributes={'version':version},error_type=None)])


class PausingPool:
    def __init__(self,pool):
        self.pool=pool;self.read=asyncio.Event();self.resume=asyncio.Event()

    @asynccontextmanager
    async def acquire(self):
        async with self.pool.acquire(timeout=2) as connection:
            owner=self
            class Connection:
                def transaction(self,**kwargs):return connection.transaction(**kwargs)
                async def fetchrow(self,*args):
                    row=await connection.fetchrow(*args)
                    owner.read.set()
                    await owner.resume.wait()
                    return row
                async def fetch(self,*args):return await connection.fetch(*args)
            yield Connection()


@pytest.mark.parametrize('change',['replace','delete'])
def test_get_keeps_one_version_when_writer_commits_between_selects(change):
    async def run():
        async with asyncpg.create_pool(os.environ['TEST_DATABASE_URL'],min_size=2,max_size=2) as pool:
            identifier='snapshot-'+uuid4().hex;old=trace(identifier,'old');new=trace(identifier,'new')
            store=PostgresTraceStore(pool);await store.save(old)
            paused=PausingPool(pool);task=asyncio.create_task(PostgresTraceStore(paused).get(identifier))
            try:
                await asyncio.wait_for(paused.read.wait(),2)
                if change=='replace':await asyncio.wait_for(store.save(new),2)
                else:await pool.execute('DELETE FROM traces WHERE trace_id=$1',identifier)
                paused.resume.set()
                result=await asyncio.wait_for(task,2)
                assert result==old
                assert await store.get(identifier)==(new if change=='replace' else None)
            finally:
                paused.resume.set()
                if not task.done():task.cancel()
                await asyncio.gather(task,return_exceptions=True)
                await pool.execute('DELETE FROM traces WHERE trace_id=$1',identifier)
    asyncio.run(run())


def test_cancelled_snapshot_reader_returns_connection():
    async def run():
        async with asyncpg.create_pool(os.environ['TEST_DATABASE_URL'],min_size=2,max_size=2) as pool:
            identifier='cancel-snapshot-'+uuid4().hex;original=trace(identifier,'old')
            store=PostgresTraceStore(pool);await store.save(original)
            paused=PausingPool(pool);task=asyncio.create_task(PostgresTraceStore(paused).get(identifier))
            try:
                await asyncio.wait_for(paused.read.wait(),2)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):await asyncio.wait_for(task,2)
                assert await store.get(identifier)==original
                assert pool.get_idle_size()==2
            finally:
                paused.resume.set()
                if not task.done():task.cancel()
                await asyncio.gather(task,return_exceptions=True)
                await pool.execute('DELETE FROM traces WHERE trace_id=$1',identifier)
    asyncio.run(run())


def test_same_id_concurrent_writers_commit_one_complete_trace():
    async def run():
        async with asyncpg.create_pool(os.environ['TEST_DATABASE_URL'],min_size=2,max_size=2) as pool:
            identifier='writers-'+uuid4().hex;versions=[trace(identifier,'v'+str(i)) for i in range(8)]
            store=PostgresTraceStore(pool)
            try:
                await asyncio.gather(*(store.save(version) for version in versions))
                assert await store.get(identifier) in versions
            finally:await pool.execute('DELETE FROM traces WHERE trace_id=$1',identifier)
    asyncio.run(run())
