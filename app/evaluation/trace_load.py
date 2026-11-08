"""Opt-in local TraceStore workload. Instrumentation never wraps production pools."""
import asyncio
from collections import Counter
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import random
import re
import sys
from time import perf_counter
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import asyncpg

from app.observability.models import SpanRecord, TraceRecord
from app.observability.store import PostgresTraceStore


@dataclass
class Sample:
    index: int
    status: str = 'pending'
    error_type: str | None = None
    wall_ms: float = 0
    acquire_ms: float = 0
    hold_ms: float = 0
    release_ms: float = 0
    sql_ms: float = 0
    transaction_ms: float = 0
    sql_calls: int = 0


class MeasuredConnection:
    def __init__(self, connection, sample, clock):
        self.connection, self.sample, self.clock = connection, sample, clock

    async def _sql(self, name, *args):
        start = self.clock()
        self.sample.sql_calls += 1
        try:
            return await getattr(self.connection, name)(*args)
        finally:
            self.sample.sql_ms += (self.clock()-start)*1000

    async def execute(self, *args): return await self._sql('execute', *args)
    async def fetchrow(self, *args): return await self._sql('fetchrow', *args)
    async def fetch(self, *args): return await self._sql('fetch', *args)

    @asynccontextmanager
    async def transaction(self, **kwargs):
        transaction = self.connection.transaction(**kwargs)
        start = self.clock()
        try:
            await transaction.__aenter__()
        finally:
            self.sample.transaction_ms += (self.clock()-start)*1000
        try:
            yield
        except BaseException as error:
            start = self.clock()
            try:
                suppress = await transaction.__aexit__(type(error), error, error.__traceback__)
            finally:
                self.sample.transaction_ms += (self.clock()-start)*1000
            if not suppress:
                raise
        else:
            start = self.clock()
            try:
                await transaction.__aexit__(None, None, None)
            finally:
                self.sample.transaction_ms += (self.clock()-start)*1000


class MeasuredPool:
    def __init__(self, pool, sample, *, timeout, clock=perf_counter):
        self.pool, self.sample, self.timeout, self.clock = pool, sample, timeout, clock

    @asynccontextmanager
    async def acquire(self):
        start = self.clock()
        try:
            connection = await self.pool.acquire(timeout=self.timeout)
        finally:
            self.sample.acquire_ms += (self.clock()-start)*1000
        start = self.clock()
        try:
            yield MeasuredConnection(connection, self.sample, self.clock)
        finally:
            self.sample.hold_ms += (self.clock()-start)*1000
            start = self.clock()
            try:
                await self.pool.release(connection, timeout=self.timeout)
            finally:
                self.sample.release_ms += (self.clock()-start)*1000


def _distribution(values):
    values = sorted(values)
    if not values:
        return None
    def percentile(q):
        position = (len(values)-1)*q
        lower = math.floor(position)
        upper = math.ceil(position)
        return values[lower] + (values[upper]-values[lower])*(position-lower)
    return {'p50': percentile(.5), 'p95': percentile(.95), 'max': values[-1]}


def summarize(samples, *, elapsed_seconds):
    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
        raise ValueError('positive_elapsed_required')
    succeeded = [s for s in samples if s.status == 'ok']
    return {'attempted': len(samples), 'succeeded': len(succeeded),
        'failed': len(samples)-len(succeeded), 'elapsed_seconds': elapsed_seconds,
        'succeeded_per_second': len(succeeded)/elapsed_seconds,
        'attempted_per_second': len(samples)/elapsed_seconds,
        'latency_all_ms': _distribution([s.wall_ms for s in samples]),
        'latency_success_ms': _distribution([s.wall_ms for s in succeeded]),
        'components_all_ms': {name: _distribution([getattr(s, name) for s in samples])
                              for name in ('acquire_ms','hold_ms','release_ms','sql_ms','transaction_ms')},
        'error_types': dict(Counter(s.error_type or 'unknown' for s in samples if s.status != 'ok'))}


def validate_admin_dsn(dsn):
    try:
        parsed = urlsplit(dsn) if isinstance(dsn, str) else None
        if (parsed is None or parsed.scheme not in ('postgres','postgresql')
                or parsed.hostname not in ('localhost','127.0.0.1','::1')
                or not parsed.path.endswith('_test') or parsed.query or parsed.fragment):
            raise ValueError
        parsed.port  # Validate malformed ports before any connection.
        return parsed
    except ValueError:
        raise ValueError('local_test_database_required') from None


def _validate_workload(operations, span_count, timeout, repeats, scenarios):
    if type(operations) is not int or not 1 <= operations <= 10000:
        raise ValueError('operations_out_of_range')
    if type(span_count) is not int or not 0 <= span_count <= 32:
        raise ValueError('span_count_out_of_range')
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 60:
        raise ValueError('timeout_out_of_range')
    if type(repeats) is not int or not 1 <= repeats <= 5:
        raise ValueError('repeats_out_of_range')
    if not scenarios or len(scenarios) > 16:
        raise ValueError('scenarios_out_of_range')
    for pool_size, concurrency in scenarios:
        if (type(pool_size) is not int or not 1 <= pool_size <= 32
                or type(concurrency) is not int or not 1 <= concurrency <= 64):
            raise ValueError('scenario_out_of_range')


def synthetic_trace(identifier, span_count):
    return TraceRecord(trace_id=identifier, request_id='trace-load', status='ok', duration_ms=1,
        spans=[SpanRecord(span_id='span-'+str(i), parent_span_id=None, name='controlled.span',
            status='ok', duration_ms=1, attributes={'index': i}, error_type=None) for i in range(span_count)])


async def run_load(pool, *, seed, mode, operations, concurrency, timeout, on_sample=None):
    if mode not in ('read','write'):
        raise ValueError('unknown_mode')
    _validate_workload(operations, len(seed.spans), timeout, 1, [(1,concurrency)])
    indexes = iter(range(operations))
    samples = [None]*operations
    active = maximum = 0
    async def worker():
        nonlocal active, maximum
        for index in indexes:
            sample = Sample(index=index)
            start = perf_counter()
            active += 1
            maximum = max(maximum, active)
            try:
                async with asyncio.timeout(timeout):
                    store = PostgresTraceStore(MeasuredPool(pool, sample, timeout=timeout))
                    if mode == 'read':
                        if await store.get(seed.trace_id) != seed:
                            raise AssertionError('trace_roundtrip_mismatch')
                    else:
                        await store.save(seed.model_copy(deep=True, update={'trace_id': seed.trace_id+'-'+str(index)}))
                sample.status = 'ok'
            except asyncio.CancelledError:
                sample.status, sample.error_type = 'cancelled', 'CancelledError'
                raise
            except Exception as error:
                sample.status, sample.error_type = 'error', type(error).__name__
            finally:
                sample.wall_ms = (perf_counter()-start)*1000
                active -= 1
                samples[index] = sample
                if on_sample is not None:
                    on_sample(sample)
    start = perf_counter()
    tasks = [asyncio.create_task(worker()) for _ in range(min(operations, concurrency))]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done(): task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    return samples, {**summarize(samples, elapsed_seconds=perf_counter()-start), 'max_active': maximum}


async def _close_owned(resource, errors, key, *, pool=False):
    """Record cleanup separately; never replace the operation's primary cause."""
    try:
        if pool:
            await asyncio.wait_for(resource.close(), timeout=5)
        else:
            await resource.close(timeout=5)
    except BaseException as error:
        errors[key] = type(error).__name__
        try:
            resource.terminate()  # Only handles created by this benchmark.
        except BaseException as terminate_error:
            errors[key+'.terminate'] = type(terminate_error).__name__
            if not isinstance(terminate_error, Exception) and isinstance(error, Exception):
                return terminate_error
        return error
    return None


async def benchmark_trace_store(admin_dsn, output_dir, *, operations=100, span_count=4,
                                timeout=5.0, repeats=2, scenarios=((1,1),(1,8),(4,8),(4,16)), order_seed=20261107):
    parsed = validate_admin_dsn(admin_dsn)
    _validate_workload(operations, span_count, timeout, repeats, scenarios)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    jobs = [(repeat, mode, size, concurrency) for repeat in range(repeats)
            for mode in ('read','write') for size, concurrency in scenarios]
    random.Random(order_seed).shuffle(jobs)
    database = 'oab_trace_bench_'+uuid4().hex+'_test'
    manifest = {'started_at': datetime.now(timezone.utc).isoformat(), 'measurement_scope':'local_trace_store_not_http_agent',
        'operations_per_phase':operations, 'span_count':span_count, 'timeout_seconds':timeout,
        'repeats':repeats, 'order_seed':order_seed, 'phase_order':jobs, 'python':platform.python_version(),
        'platform':platform.system(), 'asyncpg':version('asyncpg'), 'owned_database':database,
        'source_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
                          ('app/observability/store.py','app/evaluation/trace_load.py','scripts/benchmark_trace_store.py')},
        'limitations':['Closed-loop local synthetic workload; not production capacity or Agent/HTTP latency.',
            'Pool initialization, schema/reset and warmup excluded; row corpus reset for each phase.',
            'SQL is client roundtrip including network/decode, not pure server execution.',
            'hold includes SQL and transaction control; do not add nested timings twice.',
            'Observer/JSONL overhead included in phase wall; failures and warmup are explicit.']}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    report = {'phases':[], 'success':False, 'cleanup':{'created':False,'dropped':False}, 'cleanup_errors':{}}
    admin = None
    try:
        admin = await asyncpg.connect(admin_dsn, timeout=10, command_timeout=15)
        report['server_version'] = await admin.fetchval('SHOW server_version')
        await admin.execute(f'CREATE DATABASE "{database}"')
        report['cleanup']['created'] = True
        dsn = urlunsplit(parsed._replace(path='/'+database))
        connection = await asyncpg.connect(dsn, timeout=10, command_timeout=15)
        try:
            await connection.execute((root/'app/observability/schema.sql').read_text(encoding='utf-8'))
        finally:
            primary = sys.exception()
            error = await _close_owned(connection, report['cleanup_errors'], 'schema_connection.close')
            if error is not None and not isinstance(error, Exception) and primary is None:
                raise error
        with (output/'samples.jsonl').open('x',encoding='utf-8',buffering=1) as stream:
            for phase_id, (repeat, mode, size, concurrency) in enumerate(jobs):
                pool = await asyncpg.create_pool(dsn, min_size=size, max_size=size,
                                                timeout=10, command_timeout=timeout)
                try:
                    # This handle points only to the newly owned database.
                    await pool.execute('DELETE FROM traces')
                    seed = synthetic_trace('seed-'+str(phase_id), span_count)
                    store = PostgresTraceStore(pool)
                    await store.save(seed)
                    await asyncio.gather(*(store.get(seed.trace_id) for _ in range(size)))
                    def record(sample):
                        stream.write(json.dumps({'phase_id':phase_id, **asdict(sample)})+'\n')
                    samples, summary = await run_load(pool, seed=seed, mode=mode, operations=operations,
                        concurrency=concurrency, timeout=timeout, on_sample=record)
                    rows = await pool.fetchval('SELECT count(*) FROM traces')
                    spans = await pool.fetchval('SELECT count(*) FROM spans')
                    expected = 1+(summary['succeeded'] if mode == 'write' else 0)
                    summary.update(phase_id=phase_id, repeat=repeat, mode=mode, pool_size=size,
                        concurrency=concurrency, warmup_reads=size,
                        row_integrity={'trace_rows':rows,'span_rows':spans,
                                       'matches_acknowledged':rows==expected and spans==expected*span_count})
                    report['phases'].append(summary)
                finally:
                    primary = sys.exception()
                    error = await _close_owned(pool, report['cleanup_errors'], f'phase_{phase_id}.pool.close', pool=True)
                    if error is not None and not isinstance(error, Exception) and primary is None:
                        raise error
    except BaseException as error:
        report['failure_type'] = type(error).__name__
        raise
    finally:
        primary = sys.exception()
        cleanup_signal = None
        if admin is not None:
            try:
                if report['cleanup']['created']:
                    assert re.fullmatch(r'oab_trace_bench_[0-9a-f]{32}_test', database)
                    await admin.execute(f'DROP DATABASE "{database}"')
                    report['cleanup']['dropped'] = True
            except BaseException as error:
                report['cleanup']['error_type'] = type(error).__name__
                report['cleanup_errors']['database.drop'] = type(error).__name__
                if not isinstance(error, Exception):
                    cleanup_signal = error
            finally:
                error = await _close_owned(admin, report['cleanup_errors'], 'admin.close')
                if error is not None and not isinstance(error, Exception) and cleanup_signal is None:
                    cleanup_signal = error
        report['success'] = ('failure_type' not in report and not report['cleanup_errors'] and report['cleanup']['dropped'] and
            len(report['phases'])==len(jobs) and all(p['failed']==0 and p['row_integrity']['matches_acknowledged']
                                                   for p in report['phases']))
        (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        if cleanup_signal is not None and primary is None:
            raise cleanup_signal
    return report
