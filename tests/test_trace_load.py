"""Accounting and opt-in safety independent of a database or paid providers."""
import asyncio
from contextlib import asynccontextmanager
import importlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


def module():
    path=Path(__file__).parents[1]/'app/evaluation/trace_load.py'
    assert path.exists(),'Trace load measurement is not implemented'
    return importlib.import_module('app.evaluation.trace_load')


class Clock:
    now=0.0
    def __call__(self):return self.now
    def advance(self,ms):self.now+=ms/1000


class RawPool:
    def __init__(self,clock,fail=False):self.clock=clock;self.fail=fail;self.released=False
    async def acquire(self,*,timeout):self.clock.advance(2);return self
    async def release(self,connection,*,timeout):self.clock.advance(3);self.released=True
    @asynccontextmanager
    async def transaction(self,**kwargs):
        self.clock.advance(7)
        try:yield
        finally:self.clock.advance(11)
    async def execute(self,*args):
        self.clock.advance(5)
        if self.fail:raise RuntimeError('private connection detail')
        return 'OK'


def test_timing_does_not_add_nested_sql_to_hold_twice():
    mod=module();clock=Clock();sample=mod.Sample(index=0);raw=RawPool(clock)
    async def run():
        async with mod.MeasuredPool(raw,sample,timeout=1,clock=clock).acquire() as connection:
            async with connection.transaction():await connection.execute('SELECT 1')
    asyncio.run(run())
    assert sample.acquire_ms==pytest.approx(2)
    assert sample.hold_ms==pytest.approx(23)
    assert sample.release_ms==pytest.approx(3)
    assert sample.sql_ms==pytest.approx(5)
    assert sample.transaction_ms==pytest.approx(18)
    assert sample.sql_calls==1 and raw.released


def test_failed_query_is_timed_and_connection_returned():
    mod=module();clock=Clock();sample=mod.Sample(index=0);raw=RawPool(clock,fail=True)
    async def run():
        with pytest.raises(RuntimeError):
            async with mod.MeasuredPool(raw,sample,timeout=1,clock=clock).acquire() as connection:
                async with connection.transaction():await connection.execute('SELECT 1')
    asyncio.run(run())
    assert raw.released and sample.sql_calls==1
    assert sample.sql_ms==pytest.approx(5) and sample.transaction_ms==pytest.approx(18)


def test_cancellation_propagates_through_measurement_wrapper():
    mod=module()
    async def run():
        clock=Clock();raw=RawPool(clock);sample=mod.Sample(index=0)
        entered=asyncio.Event()
        async def wait():
            async with mod.MeasuredPool(raw,sample,timeout=1,clock=clock).acquire():
                entered.set();await asyncio.Event().wait()
        task=asyncio.create_task(wait())
        try:
            await asyncio.wait_for(entered.wait(),1);task.cancel()
            with pytest.raises(asyncio.CancelledError):await task
        finally:
            if not task.done():task.cancel()
            await asyncio.gather(task,return_exceptions=True)
        assert raw.released
    asyncio.run(run())


def test_summary_keeps_failures_and_uses_linear_percentiles():
    mod=module()
    samples=[mod.Sample(index=i,status='ok' if i<3 else 'error',wall_ms=i*10,error_type=None if i<3 else 'TimeoutError')
             for i in range(4)]
    result=mod.summarize(samples,elapsed_seconds=2)
    assert result['attempted']==4 and result['succeeded']==3 and result['failed']==1
    assert result['latency_all_ms']['p50']==15
    assert result['latency_all_ms']['p95']==pytest.approx(28.5)
    assert result['latency_success_ms']['p50']==10
    assert result['succeeded_per_second']==1.5
    assert result['error_types']=={'TimeoutError':1}


def test_no_success_does_not_report_zero_success_latency():
    mod=module()
    result=mod.summarize([mod.Sample(index=0,status='error',wall_ms=10,error_type='TimeoutError')],elapsed_seconds=1)
    assert result['latency_success_ms'] is None and result['succeeded_per_second']==0


@pytest.mark.parametrize('dsn',[None,'', 'postgresql://agent:x@127.0.0.1/observable_agent',
    'postgresql://agent:x@production/bench_test','postgresql://agent:x@localhost/bench_test?host=production',
    'postgresql://agent:x@localhost/bench_test#extra','https://localhost/bench_test'])
def test_benchmark_rejects_unsafe_admin_target(dsn):
    with pytest.raises(ValueError,match='local_test_database_required'):module().validate_admin_dsn(dsn)


@pytest.mark.parametrize('changes',[{'operations':0},{'scenarios':[(1,65)]},{'span_count':-1},{'timeout':float('nan')},
                                   {'repeats':0},{'scenarios':[(0,1)]},{'operations':True}])
def test_invalid_workload_is_rejected_before_io(monkeypatch,tmp_path,changes):
    mod=module()
    async def forbidden(*args,**kwargs):pytest.fail('Invalid workload cannot connect')
    monkeypatch.setattr(mod.asyncpg,'connect',forbidden)
    args={'operations':4,'span_count':1,'timeout':1,'repeats':1,'scenarios':[(1,2)]}
    args.update(changes)
    with pytest.raises(ValueError):
        asyncio.run(mod.benchmark_trace_store('postgresql://agent:x@localhost/bench_test',tmp_path/'new',**args))
    assert not (tmp_path/'new').exists()


def test_existing_output_directory_is_rejected_before_connect(monkeypatch,tmp_path):
    mod=module()
    async def forbidden(*args,**kwargs):pytest.fail('Existing output must be rejected before connecting')
    monkeypatch.setattr(mod.asyncpg,'connect',forbidden)
    with pytest.raises(FileExistsError):
        asyncio.run(mod.benchmark_trace_store('postgresql://agent:x@localhost/bench_test',tmp_path))


def test_cli_requires_explicit_live_before_calling_benchmark(monkeypatch,tmp_path):
    root=Path(__file__).parents[1]
    spec=importlib.util.spec_from_file_location('trace_benchmark_cli',root/'scripts/benchmark_trace_store.py')
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    async def forbidden(*args,**kwargs):pytest.fail('No --live means no benchmark call')
    monkeypatch.setattr(cli,'benchmark_trace_store',forbidden)
    monkeypatch.setattr(sys,'argv',['benchmark_trace_store.py','--output',str(tmp_path/'new')])
    with pytest.raises(SystemExit) as error:asyncio.run(cli.main())
    assert error.value.code==2 and not (tmp_path/'new').exists()


def test_cli_does_not_print_rejected_database_credentials(tmp_path):
    root=Path(__file__).parents[1];env=os.environ.copy()
    env['TRACE_BENCH_DATABASE_URL']='postgresql://user:private-cli-password@production/bench_test'
    result=subprocess.run([sys.executable,'-B',str(root/'scripts/benchmark_trace_store.py'),
        '--live','--output',str(tmp_path/'new')],env=env,capture_output=True,text=True)
    assert result.returncode==2
    assert 'private-cli-password' not in result.stdout+result.stderr
    assert 'ValueError' in result.stderr and not (tmp_path/'new').exists()


def test_unknown_mode_is_rejected_before_pool_use():
    mod=module()
    with pytest.raises(ValueError,match='unknown_mode'):
        asyncio.run(mod.run_load(None,seed=mod.synthetic_trace('seed',0),mode='unknown',
                                operations=1,concurrency=1,timeout=1))
