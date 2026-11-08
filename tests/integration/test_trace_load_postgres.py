"""Real timing paths, safe disposable database cleanup, and failure retention."""
import asyncio
import json
import os
from pathlib import Path
import re
from uuid import uuid4

import asyncpg
import pytest

from app.evaluation import trace_load as bench
from app.observability.store import PostgresTraceStore

pytestmark=pytest.mark.postgres


async def cleanup_owned_if_present(output):
    manifest=output/'manifest.json'
    if not manifest.exists():return
    name=json.loads(manifest.read_text(encoding='utf-8'))['owned_database']
    assert re.fullmatch(r'oab_trace_bench_[0-9a-f]{32}_test',name)
    connection=await asyncpg.connect(os.environ['TEST_DATABASE_URL'])
    try:
        if await connection.fetchval('SELECT 1 FROM pg_database WHERE datname=$1',name):
            await connection.execute(f'DROP DATABASE "{name}"')
    finally:await connection.close()


def test_real_benchmark_records_read_write_and_drops_own_database(tmp_path):
    async def run():
        output=tmp_path/'results'
        try:
            report=await bench.benchmark_trace_store(os.environ['TEST_DATABASE_URL'],output,
                operations=4,repeats=1,span_count=2,scenarios=[(1,2)])
            assert report['success'] and report['cleanup']=={'created':True,'dropped':True}
            samples=[json.loads(line) for line in (output/'samples.jsonl').read_text().splitlines()]
            assert len(samples)==8
            phases={p['phase_id']:p for p in report['phases']}
            for sample in samples:
                assert sample['status']=='ok'
                assert sample['sql_calls']==(2 if phases[sample['phase_id']]['mode']=='read' else 3)
                assert sample['wall_ms']+1e-6 >= sample['acquire_ms']+sample['hold_ms']+sample['release_ms']
                assert sample['hold_ms']+1e-6 >= sample['sql_ms']+sample['transaction_ms']
            assert all(p['row_integrity']['matches_acknowledged'] for p in report['phases'])
            text=''.join(path.read_text() for path in output.iterdir())
            assert 'agent_dev_password' not in text
        finally:await cleanup_owned_if_present(output)
    asyncio.run(run())


def test_real_exhausted_pool_is_counted_as_failure_then_recovers():
    async def run():
        async with asyncpg.create_pool(os.environ['TEST_DATABASE_URL'],min_size=1,max_size=1) as pool:
            seed=bench.synthetic_trace('bench-timeout-'+uuid4().hex,1)
            await PostgresTraceStore(pool).save(seed)
            try:
                async with pool.acquire(timeout=1):
                    samples,summary=await bench.run_load(pool,seed=seed,mode='read',operations=2,concurrency=2,timeout=.03)
                assert summary['failed']==2 and summary['succeeded']==0
                assert all(s.error_type=='TimeoutError' and s.sql_calls==0 and s.acquire_ms>0 for s in samples)
                _,summary=await bench.run_load(pool,seed=seed,mode='read',operations=2,concurrency=2,timeout=1)
                assert summary['succeeded']==2 and pool.get_idle_size()==1
            finally:await pool.execute('DELETE FROM traces WHERE trace_id=$1',seed.trace_id)
    asyncio.run(run())


def test_missing_trace_is_not_counted_as_success():
    async def run():
        async with asyncpg.create_pool(os.environ['TEST_DATABASE_URL'],min_size=1,max_size=1) as pool:
            seed=bench.synthetic_trace('missing-'+uuid4().hex,1)
            samples,summary=await bench.run_load(pool,seed=seed,mode='read',operations=3,concurrency=2,timeout=1)
            assert summary['failed']==3 and summary['succeeded']==0
            assert all(s.error_type=='AssertionError' and s.sql_calls==1 for s in samples)
            assert pool.get_idle_size()==1
    asyncio.run(run())


def test_benchmark_failure_still_drops_owned_database_and_redacts_error(monkeypatch,tmp_path):
    async def fail(*args,**kwargs):raise RuntimeError('private database connection details')
    monkeypatch.setattr(bench,'run_load',fail)
    async def run():
        output=tmp_path/'failure'
        try:
            with pytest.raises(RuntimeError):
                await bench.benchmark_trace_store(os.environ['TEST_DATABASE_URL'],output,
                    operations=1,repeats=1,scenarios=[(1,1)])
            report=json.loads((output/'report.json').read_text())
            assert report['cleanup']=={'created':True,'dropped':True}
            assert report['failure_type']=='RuntimeError' and not report['success']
            assert 'private database connection details' not in (output/'report.json').read_text()
        finally:await cleanup_owned_if_present(output)
    asyncio.run(run())


def test_cancelled_benchmark_cleans_up_and_preserves_cancellation(monkeypatch,tmp_path):
    async def run():
        entered=asyncio.Event()
        async def wait(*args,**kwargs):entered.set();await asyncio.Event().wait()
        monkeypatch.setattr(bench,'run_load',wait)
        output=tmp_path/'cancelled'
        task=asyncio.create_task(bench.benchmark_trace_store(os.environ['TEST_DATABASE_URL'],output,
            operations=1,repeats=1,scenarios=[(1,1)]))
        try:
            await asyncio.wait_for(entered.wait(),3);task.cancel()
            with pytest.raises(asyncio.CancelledError):await asyncio.wait_for(task,3)
            report=json.loads((output/'report.json').read_text())
            assert report['cleanup']=={'created':True,'dropped':True}
            assert report['failure_type']=='CancelledError' and not report['success']
        finally:
            if not task.done():task.cancel()
            await asyncio.gather(task,return_exceptions=True)
            await cleanup_owned_if_present(output)
    asyncio.run(run())


@pytest.mark.parametrize('resource,primary', [('admin','error'),('pool','error'),
    ('admin','cancel'),('pool','cancel'),('schema','schema_error')])
def test_cleanup_error_preserves_primary_failure_and_report(monkeypatch,tmp_path,resource,primary):
    async def run():
        output=tmp_path/(resource+'-'+primary)
        original_connect=asyncpg.connect;original_pool=asyncpg.create_pool
        entered=asyncio.Event()
        class CloseFailure:
            def __init__(self,owned):self.owned=owned
            def __getattr__(self,name):return getattr(self.owned,name)
            async def close(self,*args,**kwargs):
                await self.owned.close(*args,**kwargs)
                raise TimeoutError('private cleanup details')
            async def execute(self,query,*args):
                if resource=='schema' and 'CREATE TABLE' in query:
                    raise ValueError('primary schema failure')
                return await self.owned.execute(query,*args)
        async def connect(dsn,**kwargs):
            conn=await original_connect(dsn,**kwargs)
            is_admin=dsn==os.environ['TEST_DATABASE_URL']
            return CloseFailure(conn) if (resource=='admin' and is_admin or resource=='schema' and not is_admin) else conn
        async def pool(*args,**kwargs):
            value=await original_pool(*args,**kwargs)
            return CloseFailure(value) if resource=='pool' else value
        async def work(*args,**kwargs):
            if primary=='error':raise RuntimeError('primary workload failure')
            entered.set();await asyncio.Event().wait()
        try:
            with monkeypatch.context() as patch:
                patch.setattr(bench.asyncpg,'connect',connect)
                patch.setattr(bench.asyncpg,'create_pool',pool)
                patch.setattr(bench,'run_load',work)
                if primary=='cancel':
                    task=asyncio.create_task(bench.benchmark_trace_store(os.environ['TEST_DATABASE_URL'],output,
                        operations=1,repeats=1,scenarios=[(1,1)]))
                    try:
                        await asyncio.wait_for(entered.wait(),3);task.cancel()
                        with pytest.raises(asyncio.CancelledError):await asyncio.wait_for(task,3)
                    finally:
                        if not task.done():task.cancel()
                        await asyncio.gather(task,return_exceptions=True)
                else:
                    with pytest.raises(ValueError if primary=='schema_error' else RuntimeError):
                        await bench.benchmark_trace_store(os.environ['TEST_DATABASE_URL'],output,
                            operations=1,repeats=1,scenarios=[(1,1)])
                report=json.loads((output/'report.json').read_text())
                assert report['failure_type']=={'error':'RuntimeError','cancel':'CancelledError','schema_error':'ValueError'}[primary]
                assert 'TimeoutError' in report['cleanup_errors'].values()
                assert report['cleanup']['dropped'] and not report['success']
                assert 'private cleanup details' not in (output/'report.json').read_text()
        finally:await cleanup_owned_if_present(output)
    asyncio.run(run())
