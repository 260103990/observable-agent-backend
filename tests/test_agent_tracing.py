import asyncio
import pytest
import logging

from app.agent.runner import AgentRunner
from app.observability.store import InMemoryTraceStore
from app.schemas.agent import ToolCall
from app.tools.calculator import ADD_NUMBERS_TOOL
from app.tools.registry import ToolRegistry


def test_runner_records_successful_trace():
    async def fake_decision_maker(prompt, tool_schemas):
        return ToolCall(
            name="add_numbers",
            arguments={"a": 17, "b": 25},
        )

    async def run_test():
        registry = ToolRegistry()
        registry.register(ADD_NUMBERS_TOOL)

        store = InMemoryTraceStore()

        runner = AgentRunner(
            registry=registry,
            decision_maker=fake_decision_maker,
            trace_store=store,
        )

        result = await runner.run(
            "请计算 17 + 25",
            request_id="request-runner-123",
            trace_id="trace-runner-123",
        )

        trace = await store.get("trace-runner-123")

        return result, trace

    result, trace = asyncio.run(run_test())

    assert result == 42
    assert trace is not None
    assert trace.request_id == "request-runner-123"
    assert trace.trace_id == "trace-runner-123"
    assert trace.status == "ok"

    root_span = trace.spans[0]

    assert root_span.name == "agent.run"
    assert root_span.parent_span_id is None
    assert root_span.status == "ok"
    assert root_span.duration_ms >= 0
    assert len(trace.spans) >= 2

    model_span = trace.spans[1]

    assert model_span.name == "model.decision"
    assert model_span.span_id != root_span.span_id
    assert model_span.parent_span_id == root_span.span_id
    assert model_span.status == "ok"
    assert model_span.duration_ms >= 0
    assert model_span.error_type is None
    assert model_span.attributes == {
        "available_tool_count": 1,
    }

    assert len(trace.spans) == 3

    tool_span = trace.spans[2]

    assert tool_span.name == "tool.execute"
    assert tool_span.span_id != root_span.span_id
    assert tool_span.span_id != model_span.span_id
    assert tool_span.parent_span_id == root_span.span_id
    assert tool_span.status == "ok"
    assert tool_span.duration_ms >= 0
    assert tool_span.error_type is None
    assert tool_span.attributes == {
        "tool_name": "add_numbers",
    }
def test_runner_records_model_failure():
    failure = RuntimeError("模拟模型决策失败")

    async def fake_decision_maker(prompt, tool_schemas):
        raise failure

    async def run_test():
        registry = ToolRegistry()
        registry.register(ADD_NUMBERS_TOOL)

        store = InMemoryTraceStore()

        runner = AgentRunner(
            registry=registry,
            decision_maker=fake_decision_maker,
            trace_store=store,
        )

        with pytest.raises(RuntimeError) as exc_info:
            await runner.run(
                "请计算 17 + 25",
                request_id="request-model-failed",
                trace_id="trace-model-failed",
            )

        assert exc_info.value is failure

        return await store.get("trace-model-failed")

    trace = asyncio.run(run_test())

    assert trace is not None
    assert trace.status == "error"

    assert [span.name for span in trace.spans] == [
        "agent.run",
        "model.decision",
    ]

    root_span, model_span = trace.spans

    assert root_span.parent_span_id is None
    assert model_span.parent_span_id == root_span.span_id

    for span in trace.spans:
        assert span.status == "error"
        assert span.error_type == "RuntimeError"
        assert span.duration_ms >= 0

def test_runner_records_tool_failure(monkeypatch):
    failure = RuntimeError("模拟工具执行失败")

    async def fake_decision_maker(prompt, tool_schemas):
        return ToolCall(
            name="add_numbers",
            arguments={"a": 17, "b": 25},
        )

    def fake_execute(name, arguments):
        raise failure

    async def run_test():
        registry = ToolRegistry()
        registry.register(ADD_NUMBERS_TOOL)

        monkeypatch.setattr(
            registry,
            "execute",
            fake_execute,
        )

        store = InMemoryTraceStore()

        runner = AgentRunner(
            registry=registry,
            decision_maker=fake_decision_maker,
            trace_store=store,
        )

        with pytest.raises(RuntimeError) as exc_info:
            await runner.run(
                "请计算 17 + 25",
                request_id="request-tool-failed",
                trace_id="trace-tool-failed",
            )

        assert exc_info.value is failure

        return await store.get("trace-tool-failed")

    trace = asyncio.run(run_test())

    assert trace is not None
    assert trace.status == "error"

    assert [span.name for span in trace.spans] == [
        "agent.run",
        "model.decision",
        "tool.execute",
    ]

    root_span, model_span, tool_span = trace.spans

    assert model_span.status == "ok"
    assert model_span.error_type is None

    assert model_span.parent_span_id == root_span.span_id
    assert tool_span.parent_span_id == root_span.span_id

    for span in (root_span, tool_span):
        assert span.status == "error"
        assert span.error_type == "RuntimeError"
        assert span.duration_ms >= 0

    assert tool_span.attributes == {
        "tool_name": "add_numbers",
    }

class FailingTraceStore:
    def __init__(self):
        self.save_calls = 0

    async def save(self, trace):
        self.save_calls += 1
        raise RuntimeError("模拟 Trace 保存失败")

    async def get(self, trace_id):
        return None


def test_runner_returns_result_when_trace_save_fails():
    store = FailingTraceStore()

    async def fake_decision_maker(prompt, tool_schemas):
        return ToolCall(
            name="add_numbers",
            arguments={"a": 17, "b": 25},
        )

    async def run_test():
        registry = ToolRegistry()
        registry.register(ADD_NUMBERS_TOOL)

        runner = AgentRunner(
            registry=registry,
            decision_maker=fake_decision_maker,
            trace_store=store,
        )

        return await runner.run(
            "请计算 17 + 25",
            request_id="request-save-failed",
            trace_id="trace-save-failed",
        )

    result = asyncio.run(run_test())

    assert result == 42
    assert store.save_calls == 1


@pytest.mark.parametrize("failure_stage", ["model", "tool"])
def test_runner_preserves_business_error_when_trace_save_fails(
    monkeypatch,
    failure_stage,
):
    failure = RuntimeError("原来的业务异常")
    store = FailingTraceStore()

    async def fake_decision_maker(prompt, tool_schemas):
        if failure_stage == "model":
            raise failure

        return ToolCall(
            name="add_numbers",
            arguments={"a": 17, "b": 25},
        )

    def fake_execute(name, arguments):
        raise failure

    async def run_test():
        registry = ToolRegistry()
        registry.register(ADD_NUMBERS_TOOL)

        monkeypatch.setattr(
            registry,
            "execute",
            fake_execute,
        )

        runner = AgentRunner(
            registry=registry,
            decision_maker=fake_decision_maker,
            trace_store=store,
        )

        with pytest.raises(RuntimeError) as exc_info:
            await runner.run(
                "请计算 17 + 25",
                request_id="request-business-save-failed",
                trace_id="trace-business-save-failed",
            )

        return exc_info.value

    received_error = asyncio.run(run_test())

    assert received_error is failure
    assert store.save_calls == 1


def test_runner_logs_trace_save_failure(caplog):
    store = FailingTraceStore()

    async def fake_decision_maker(prompt, tool_schemas):
        return ToolCall(
            name="add_numbers",
            arguments={"a": 17, "b": 25},
        )

    async def run_test():
        registry = ToolRegistry()
        registry.register(ADD_NUMBERS_TOOL)

        runner = AgentRunner(
            registry=registry,
            decision_maker=fake_decision_maker,
            trace_store=store,
        )

        return await runner.run(
            "请计算 17 + 25",
            request_id="request-save-log",
            trace_id="trace-save-log",
        )

    with caplog.at_level(
        logging.WARNING,
        logger="app.agent",
    ):
        result = asyncio.run(run_test())

    assert result == 42
    assert store.save_calls == 1

    records = [
        record
        for record in caplog.records
        if record.name == "app.agent"
        and record.getMessage() == "trace_save_failed"
    ]

    assert len(records) == 1

    record = records[0]

    assert record.levelno == logging.WARNING
    assert record.request_id == "request-save-log"
    assert record.trace_id == "trace-save-log"
    assert record.exception_type == "RuntimeError"
    assert record.exc_info is None

    assert "模拟 Trace 保存失败" not in caplog.text
    assert "请计算 17 + 25" not in caplog.text
