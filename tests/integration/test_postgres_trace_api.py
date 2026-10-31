import asyncio
import os
from uuid import uuid4

import asyncpg
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.observability.models import SpanRecord, TraceRecord
from app.observability.store import PostgresTraceStore
from app.api import dependencies
from app.schemas.agent import ToolCall, ToolDecision


pytestmark = pytest.mark.postgres


def test_trace_api_uses_postgres_store(monkeypatch):
    database_url = os.environ["TEST_DATABASE_URL"]

    trace = TraceRecord(
        trace_id=f"trace-api-{uuid4()}",
        request_id="request-api-postgres",
        status="ok",
        duration_ms=1.0,
        spans=[
            SpanRecord(
                span_id="span-api",
                parent_span_id=None,
                name="tool.call",
                status="ok",
                duration_ms=1.0,
                attributes={"source": "postgres"},
                error_type=None,
            )
        ],
    )

    async def save_trace():
        async with asyncpg.create_pool(
            dsn=database_url,
            min_size=1,
            max_size=1,
        ) as pool:
            writer = PostgresTraceStore(pool)
            await writer.save(trace)

    async def delete_trace():
        async with asyncpg.create_pool(
            dsn=database_url,
            min_size=1,
            max_size=1,
        ) as pool:
            async with pool.acquire() as connection:
                await connection.execute(
                    "DELETE FROM traces WHERE trace_id = $1",
                    trace.trace_id,
                )

    monkeypatch.setenv("DATABASE_URL", database_url)
    get_settings.cache_clear()

    asyncio.run(save_trace())

    try:
        with TestClient(app) as client:
            response = client.get(
                f"/traces/{trace.trace_id}"
            )
            missing_response = client.get(
                f"/traces/missing-{uuid4()}"
            )

        assert response.status_code == 200
        assert response.json() == trace.model_dump(
            mode="json"
        )

        assert missing_response.status_code == 404
        assert missing_response.json() == {
            "detail": "Trace not found",
        }
    finally:
        get_settings.cache_clear()
        asyncio.run(delete_trace())


def test_agent_api_saves_trace_to_postgres(monkeypatch):
    database_url = os.environ["TEST_DATABASE_URL"]
    request_id = f"request-agent-postgres-{uuid4()}"
    trace_id = None

    class FakeDecisionMaker:
        def __init__(self, **kwargs):
            pass

        async def __call__(self, prompt, tool_schemas):
            tool_call = ToolCall(
                name="add_numbers",
                arguments={"a": 17, "b": 25},
            )

            return ToolDecision(
                tool_call=tool_call,
                assistant_message={
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": tool_call.model_dump()},
                    ],
                },
            )

    class FakeSummarizer:
        def __init__(self, **kwargs):
            pass

        async def __call__(self, prompt, decision, tool_result):
            assert prompt == "请计算 17 + 25"
            assert tool_result == 42
            return "结果是 42。"

    async def read_and_delete_trace(trace_id):
        async with asyncpg.create_pool(
            dsn=database_url,
            min_size=1,
            max_size=1,
        ) as pool:
            try:
                if trace_id is None:
                    return None

                reader = PostgresTraceStore(pool)
                return await reader.get(trace_id)
            finally:
                async with pool.acquire() as connection:
                    await connection.execute(
                        "DELETE FROM traces WHERE request_id = $1",
                        request_id,
                    )

    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setattr(
        dependencies,
        "OllamaDecisionMaker",
        FakeDecisionMaker,
    )
    monkeypatch.setattr(
        dependencies,
        "OllamaSummarizer",
        FakeSummarizer,
    )
    get_settings.cache_clear()

    try:
        with TestClient(app) as client:
            response = client.post(
                "/agent/run",
                headers={"X-Request-ID": request_id},
                json={"prompt": "请计算 17 + 25"},
            )

            assert response.status_code == 200
            assert response.json() == {
                "result": 42,
                "answer": "结果是 42。",
            }

            trace_id = response.headers["X-Trace-ID"]
            trace_response = client.get(f"/traces/{trace_id}")

            assert trace_response.status_code == 200
    finally:
        get_settings.cache_clear()
        stored_trace = asyncio.run(
            read_and_delete_trace(trace_id)
        )

    assert stored_trace is not None
    assert stored_trace.trace_id == trace_id
    assert stored_trace.request_id == request_id
    assert stored_trace.status == "ok"

    assert [span.name for span in stored_trace.spans] == [
        "agent.run",
        "model.decision",
        "tool.execute",
        "model.summary",
    ]
    assert all(
        span.status == "ok"
        for span in stored_trace.spans
    )

    assert trace_response.json() == stored_trace.model_dump(
        mode="json"
    )
