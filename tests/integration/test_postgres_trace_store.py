import asyncio
import os
from importlib import import_module
from uuid import uuid4

import asyncpg
import pytest


pytestmark = pytest.mark.postgres


def test_postgres_trace_store_returns_none_for_unknown_id():
    stores = import_module("app.observability.store")

    assert hasattr(
        stores,
        "PostgresTraceStore",
    ), "尚未定义 PostgresTraceStore"

    async def run_test():
        async with asyncpg.create_pool(
            dsn=os.environ["TEST_DATABASE_URL"],
            min_size=1,
            max_size=1,
        ) as pool:
            store = stores.PostgresTraceStore(pool)
            trace_id = f"missing-{uuid4()}"

            return await store.get(trace_id)

    result = asyncio.run(run_test())

    assert result is None


def test_postgres_trace_store_saves_and_gets_trace_without_spans():
    models = import_module("app.observability.models")
    stores = import_module("app.observability.store")

    trace = models.TraceRecord(
        trace_id=f"trace-postgres-{uuid4()}",
        request_id="request-postgres-save",
        status="ok",
        duration_ms=12.5,
        spans=[],
    )

    async def run_test():
        async with asyncpg.create_pool(
            dsn=os.environ["TEST_DATABASE_URL"],
            min_size=1,
            max_size=1,
        ) as pool:
            writer = stores.PostgresTraceStore(pool)
            await writer.save(trace)

            try:
                reader = stores.PostgresTraceStore(pool)
                return await reader.get(trace.trace_id)
            finally:
                async with pool.acquire() as connection:
                    await connection.execute(
                        "DELETE FROM traces WHERE trace_id = $1",
                        trace.trace_id,
                    )

    stored_trace = asyncio.run(run_test())

    assert stored_trace == trace


def test_postgres_trace_store_saves_and_gets_trace_with_one_span():
    models = import_module("app.observability.models")
    stores = import_module("app.observability.store")

    trace = models.TraceRecord(
        trace_id=f"trace-postgres-{uuid4()}",
        request_id="request-postgres-span",
        status="ok",
        duration_ms=12.5,
        spans=[
            models.SpanRecord(
                span_id="span-postgres-save",
                parent_span_id=None,
                name="tool.call",
                status="ok",
                duration_ms=3.5,
                attributes={
                    "tool_name": "calculator",
                    "attempt": 1,
                    "cached": False,
                    "score": 0.5,
                },
                error_type=None,
            )
        ],
    )

    async def run_test():
        async with asyncpg.create_pool(
            dsn=os.environ["TEST_DATABASE_URL"],
            min_size=1,
            max_size=1,
        ) as pool:
            writer = stores.PostgresTraceStore(pool)
            await writer.save(trace)

            try:
                reader = stores.PostgresTraceStore(pool)
                return await reader.get(trace.trace_id)
            finally:
                async with pool.acquire() as connection:
                    await connection.execute(
                        "DELETE FROM traces WHERE trace_id = $1",
                        trace.trace_id,
                    )

    stored_trace = asyncio.run(run_test())

    assert stored_trace == trace


def test_postgres_trace_store_restores_span_order():
    models = import_module("app.observability.models")
    stores = import_module("app.observability.store")

    trace = models.TraceRecord(
        trace_id=f"trace-postgres-{uuid4()}",
        request_id="request-postgres-order",
        status="ok",
        duration_ms=2.0,
        spans=[
            models.SpanRecord(
                span_id=span_id,
                parent_span_id=None,
                name=span_id,
                status="ok",
                duration_ms=1.0,
                attributes={},
                error_type=None,
            )
            for span_id in ("span-z", "span-a")
        ],
    )

    async def run_test():
        async with asyncpg.create_pool(
            dsn=os.environ["TEST_DATABASE_URL"],
            min_size=1,
            max_size=1,
            server_settings={
                "enable_indexscan": "off",
                "enable_bitmapscan": "off",
            },
        ) as pool:
            writer = stores.PostgresTraceStore(pool)
            await writer.save(trace)

            try:
                async with pool.acquire() as connection:
                    await connection.execute(
                        "DELETE FROM spans WHERE trace_id = $1",
                        trace.trace_id,
                    )

                    await connection.execute(
                        """
                        INSERT INTO spans (
                            trace_id,
                            span_index,
                            span_id,
                            parent_span_id,
                            name,
                            status,
                            duration_ms,
                            attributes,
                            error_type
                        )
                        VALUES
                            ($1, 1, $2, NULL, $2, 'ok', 1.0, '{}', NULL),
                            ($1, 0, $3, NULL, $3, 'ok', 1.0, '{}', NULL)
                        """,
                        trace.trace_id,
                        trace.spans[1].span_id,
                        trace.spans[0].span_id,
                    )

                    raw_rows = await connection.fetch(
                        """
                        SELECT span_index
                        FROM spans
                        WHERE trace_id = $1
                        """,
                        trace.trace_id,
                    )

                    assert [
                        row["span_index"] for row in raw_rows
                    ] == [1, 0], "测试夹具未形成反序，请贴出失败信息"

                reader = stores.PostgresTraceStore(pool)
                return await reader.get(trace.trace_id)
            finally:
                async with pool.acquire() as connection:
                    await connection.execute(
                        "DELETE FROM traces WHERE trace_id = $1",
                        trace.trace_id,
                    )

    stored_trace = asyncio.run(run_test())

    assert stored_trace == trace


def test_postgres_trace_store_replaces_trace_with_same_id():
    models = import_module("app.observability.models")
    stores = import_module("app.observability.store")

    trace_id = f"trace-postgres-{uuid4()}"

    original = models.TraceRecord(
        trace_id=trace_id,
        request_id="request-before",
        status="ok",
        duration_ms=2.0,
        spans=[
            models.SpanRecord(
                span_id=span_id,
                parent_span_id=None,
                name=span_id,
                status="ok",
                duration_ms=1.0,
                attributes={},
                error_type=None,
            )
            for span_id in ("span-keep", "span-remove")
        ],
    )

    updated = models.TraceRecord(
        trace_id=trace_id,
        request_id="request-after",
        status="error",
        duration_ms=3.5,
        spans=[
            models.SpanRecord(
                span_id="span-keep",
                parent_span_id=None,
                name="retry",
                status="error",
                duration_ms=3.5,
                attributes={"attempt": 2},
                error_type="ToolError",
            )
        ],
    )

    async def run_test():
        async with asyncpg.create_pool(
            dsn=os.environ["TEST_DATABASE_URL"],
            min_size=1,
            max_size=1,
        ) as pool:
            writer = stores.PostgresTraceStore(pool)
            await writer.save(original)

            try:
                await writer.save(updated)

                reader = stores.PostgresTraceStore(pool)
                return await reader.get(trace_id)
            finally:
                async with pool.acquire() as connection:
                    await connection.execute(
                        "DELETE FROM traces WHERE trace_id = $1",
                        trace_id,
                    )

    stored_trace = asyncio.run(run_test())

    assert stored_trace == updated


def test_postgres_trace_store_preserves_original_when_save_fails(
    monkeypatch,
):
    models = import_module("app.observability.models")
    stores = import_module("app.observability.store")

    def make_span(span_id, attributes):
        return models.SpanRecord(
            span_id=span_id,
            parent_span_id=None,
            name=span_id,
            status="ok",
            duration_ms=1.0,
            attributes=attributes,
            error_type=None,
        )

    trace_id = f"trace-postgres-{uuid4()}"

    original = models.TraceRecord(
        trace_id=trace_id,
        request_id="request-original",
        status="ok",
        duration_ms=1.0,
        spans=[make_span("span-original", {})],
    )

    updated = models.TraceRecord(
        trace_id=trace_id,
        request_id="request-updated",
        status="error",
        duration_ms=2.0,
        spans=[
            make_span("span-new-1", {}),
            make_span("span-new-2", {"force_failure": True}),
        ],
    )

    real_dumps = stores.json.dumps

    def fail_dumps(attributes):
        if attributes == {"force_failure": True}:
            raise RuntimeError("forced serialization failure")
        return real_dumps(attributes)

    async def run_test():
        async with asyncpg.create_pool(
            dsn=os.environ["TEST_DATABASE_URL"],
            min_size=1,
            max_size=1,
        ) as pool:
            writer = stores.PostgresTraceStore(pool)
            await writer.save(original)

            try:
                with monkeypatch.context() as patch:
                    patch.setattr(stores.json, "dumps", fail_dumps)

                    with pytest.raises(
                        RuntimeError,
                        match="forced serialization failure",
                    ):
                        await writer.save(updated)

                reader = stores.PostgresTraceStore(pool)
                return await reader.get(trace_id)
            finally:
                async with pool.acquire() as connection:
                    await connection.execute(
                        "DELETE FROM traces WHERE trace_id = $1",
                        trace_id,
                    )

    stored_trace = asyncio.run(run_test())

    assert stored_trace == original


def test_postgres_trace_store_replaces_spans_with_empty_list():
    models = import_module("app.observability.models")
    stores = import_module("app.observability.store")

    original = models.TraceRecord(
        trace_id=f"trace-postgres-{uuid4()}",
        request_id="request-clear-spans",
        status="ok",
        duration_ms=2.0,
        spans=[
            models.SpanRecord(
                span_id=span_id,
                parent_span_id=None,
                name=span_id,
                status="ok",
                duration_ms=1.0,
                attributes={},
                error_type=None,
            )
            for span_id in ("span-first", "span-second")
        ],
    )

    updated = original.model_copy(update={"spans": []})

    async def run_test():
        async with asyncpg.create_pool(
            dsn=os.environ["TEST_DATABASE_URL"],
            min_size=1,
            max_size=1,
        ) as pool:
            writer = stores.PostgresTraceStore(pool)
            await writer.save(original)

            try:
                await writer.save(updated)

                reader = stores.PostgresTraceStore(pool)
                return await reader.get(original.trace_id)
            finally:
                async with pool.acquire() as connection:
                    await connection.execute(
                        "DELETE FROM traces WHERE trace_id = $1",
                        original.trace_id,
                    )

    stored_trace = asyncio.run(run_test())

    assert stored_trace == updated


def test_postgres_trace_store_keeps_objects_independent():
    models = import_module("app.observability.models")
    stores = import_module("app.observability.store")

    trace = models.TraceRecord(
        trace_id=f"trace-postgres-{uuid4()}",
        request_id="request-independent",
        status="ok",
        duration_ms=1.0,
        spans=[
            models.SpanRecord(
                span_id="span-independent",
                parent_span_id=None,
                name="tool.call",
                status="ok",
                duration_ms=1.0,
                attributes={"attempt": 1},
                error_type=None,
            )
        ],
    )

    expected = trace.model_copy(deep=True)

    async def run_test():
        async with asyncpg.create_pool(
            dsn=os.environ["TEST_DATABASE_URL"],
            min_size=1,
            max_size=1,
        ) as pool:
            writer = stores.PostgresTraceStore(pool)
            await writer.save(trace)

            try:
                trace.request_id = "request-input-mutated"
                trace.spans[0].attributes["attempt"] = 2
                trace.spans.clear()

                reader = stores.PostgresTraceStore(pool)
                first = await reader.get(trace.trace_id)

                assert first == expected

                first.request_id = "request-output-mutated"
                first.spans[0].attributes["attempt"] = 3
                first.spans.clear()

                return await reader.get(trace.trace_id)
            finally:
                async with pool.acquire() as connection:
                    await connection.execute(
                        "DELETE FROM traces WHERE trace_id = $1",
                        trace.trace_id,
                    )

    stored_trace = asyncio.run(run_test())

    assert stored_trace == expected


def test_postgres_trace_store_accepts_plain_and_repeated_ids():
    models = import_module("app.observability.models")
    stores = import_module("app.observability.store")

    suffix = str(uuid4())

    span = models.SpanRecord(
        span_id=f"shared-span-{suffix}",
        parent_span_id=f"external-parent-{suffix}",
        name="repeatable-span",
        status="ok",
        duration_ms=1.0,
        attributes={},
        error_type=None,
    )

    trace_a = models.TraceRecord(
        trace_id=f"trace-a-{suffix}",
        request_id=f"shared-request-{suffix}",
        status="ok",
        duration_ms=2.0,
        spans=[span, span.model_copy(deep=True)],
    )

    trace_b = trace_a.model_copy(
        deep=True,
        update={"trace_id": f"trace-b-{suffix}"},
    )

    async def run_test():
        async with asyncpg.create_pool(
            dsn=os.environ["TEST_DATABASE_URL"],
            min_size=1,
            max_size=1,
        ) as pool:
            try:
                writer = stores.PostgresTraceStore(pool)
                await writer.save(trace_a)
                await writer.save(trace_b)

                reader = stores.PostgresTraceStore(pool)
                return (
                    await reader.get(trace_a.trace_id),
                    await reader.get(trace_b.trace_id),
                )
            finally:
                async with pool.acquire() as connection:
                    await connection.execute(
                        """
                        DELETE FROM traces
                        WHERE trace_id = $1 OR trace_id = $2
                        """,
                        trace_a.trace_id,
                        trace_b.trace_id,
                    )

    loaded_a, loaded_b = asyncio.run(run_test())

    assert loaded_a == trace_a
    assert loaded_b == trace_b
