from importlib import import_module
import asyncio


def test_trace_record_represents_agent_run():
    models = import_module(
        "app.observability.models"
    )

    assert hasattr(
        models,
        "SpanRecord",
    ), "尚未定义 SpanRecord"

    assert hasattr(
        models,
        "TraceRecord",
    ), "尚未定义 TraceRecord"

    span = models.SpanRecord(
        span_id="span-root",
        parent_span_id=None,
        name="agent.run",
        status="ok",
        duration_ms=3.0,
        attributes={},
        error_type=None,
    )

    trace = models.TraceRecord(
        trace_id="trace-123",
        request_id="request-123",
        status="ok",
        duration_ms=3.0,
        spans=[span],
    )

    assert trace.trace_id == "trace-123"
    assert trace.request_id == "request-123"
    assert trace.status == "ok"
    assert trace.spans == [span]


def test_trace_store_saves_and_gets_trace():
    models = import_module(
        "app.observability.models"
    )
    stores = import_module(
        "app.observability.store"
    )

    assert hasattr(
        stores,
        "InMemoryTraceStore",
    ), "尚未定义 InMemoryTraceStore"

    trace = models.TraceRecord(
        trace_id="trace-store-123",
        request_id="request-store-123",
        status="ok",
        duration_ms=3.0,
        spans=[],
    )

    async def run_test():
        store = stores.InMemoryTraceStore()

        await store.save(trace)

        return await store.get("trace-store-123")

    stored_trace = asyncio.run(run_test())

    assert stored_trace == trace


def test_trace_store_returns_independent_copy():
    models = import_module(
        "app.observability.models"
    )
    stores = import_module(
        "app.observability.store"
    )

    span = models.SpanRecord(
        span_id="span-root",
        parent_span_id=None,
        name="agent.run",
        status="ok",
        duration_ms=3.0,
        attributes={},
        error_type=None,
    )

    trace = models.TraceRecord(
        trace_id="trace-copy-123",
        request_id="request-copy-123",
        status="ok",
        duration_ms=3.0,
        spans=[span],
    )

    async def run_test():
        store = stores.InMemoryTraceStore()
        await store.save(trace)

        first_read = await store.get("trace-copy-123")
        assert first_read is not None

        first_read.spans[0].attributes["changed"] = True

        return await store.get("trace-copy-123")

    second_read = asyncio.run(run_test())

    assert second_read is not None
    assert second_read.spans[0].attributes == {}


def test_trace_store_saves_independent_copy():
    models = import_module(
        "app.observability.models"
    )
    stores = import_module(
        "app.observability.store"
    )

    span = models.SpanRecord(
        span_id="span-root",
        parent_span_id=None,
        name="agent.run",
        status="ok",
        duration_ms=3.0,
        attributes={},
        error_type=None,
    )

    trace = models.TraceRecord(
        trace_id="trace-save-copy-123",
        request_id="request-save-copy-123",
        status="ok",
        duration_ms=3.0,
        spans=[span],
    )

    async def run_test():
        store = stores.InMemoryTraceStore()

        await store.save(trace)

        trace.spans[0].attributes["changed"] = True

        return await store.get("trace-save-copy-123")

    stored_trace = asyncio.run(run_test())

    assert stored_trace is not None
    assert stored_trace.spans[0].attributes == {}


def test_trace_store_returns_none_for_unknown_id():
    stores = import_module(
        "app.observability.store"
    )

    async def run_test():
        store = stores.InMemoryTraceStore()

        return await store.get("missing-trace")

    result = asyncio.run(run_test())

    assert result is None
