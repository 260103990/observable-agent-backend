import asyncio
from uuid import uuid4
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.agent import ToolCall, ToolDecision
from app.observability.models import TraceRecord
from app.api import dependencies
from app.observability.store import InMemoryTraceStore


def test_get_trace_store_reuses_same_instance():
    assert hasattr(
        dependencies,
        "get_trace_store",
    ), "尚未实现 get_trace_store"

    first = dependencies.get_trace_store()
    second = dependencies.get_trace_store()

    assert isinstance(first, InMemoryTraceStore)
    assert first is second


def test_trace_saved_through_one_provider_can_be_read_through_another():
    trace = TraceRecord(
        trace_id=str(uuid4()),
        request_id="request-shared-store",
        status="ok",
        duration_ms=1.0,
        spans=[],
    )

    async def run_test():
        writer = dependencies.get_trace_store()
        reader = dependencies.get_trace_store()

        await writer.save(trace)
        return await reader.get(trace.trace_id)

    saved = asyncio.run(run_test())

    assert saved == trace


def test_real_agent_runner_saves_trace_in_shared_store(
    monkeypatch,
):
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

    store = dependencies.get_trace_store()

    with TestClient(app) as client:
        response = client.post(
            "/agent/run",
            headers={
                "X-Request-ID": "request-runner-store",
            },
            json={"prompt": "请计算 17 + 25"},
        )

        assert response.status_code == 200
        assert response.json() == {
            "result": 42,
            "answer": "结果是 42。",
        }

        trace_id = response.headers["X-Trace-ID"]
        trace_response = client.get(f"/traces/{trace_id}")

    trace = asyncio.run(store.get(trace_id))

    assert trace is not None
    assert trace.trace_id == trace_id
    assert trace.request_id == "request-runner-store"
    assert trace.status == "ok"

    assert [span.name for span in trace.spans] == [
        "agent.run",
        "model.decision",
        "tool.execute",
        "model.summary",
    ]
    assert all(span.status == "ok" for span in trace.spans)

    assert trace_response.status_code == 200
    assert trace_response.json() == trace.model_dump(
        mode="json"
    )
