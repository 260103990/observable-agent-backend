import asyncio
import pytest
import logging

from app.agent.runner import AgentRunner
from app.observability.store import InMemoryTraceStore
from app.schemas.agent import ToolCall, ToolDecision
from app.tools.calculator import ADD_NUMBERS_TOOL
from app.tools.registry import ToolRegistry
from app.agent.ollama_summarizer import OllamaSummaryError


def make_tool_decision(name, arguments):
    tool_call = ToolCall(
        name=name,
        arguments=arguments,
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


async def successful_summary(prompt, decision, tool_result):
    assert tool_result == 42
    return "结果是 42。"


async def unexpected_summary(prompt, decision, tool_result):
    raise AssertionError("前序失败后不应进入总结")


def test_runner_records_successful_trace(monkeypatch):
    clock = {"now": 10.0}

    monkeypatch.setattr(
        "app.agent.runner.perf_counter",
        lambda: clock["now"],
    )

    tool_call = ToolCall(
        name="add_numbers",
        arguments={"a": 17, "b": 25},
    )

    decision = ToolDecision(
        tool_call=tool_call,
        assistant_message={
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": tool_call.model_dump()},
            ],
        },
    )

    async def fake_decision_maker(prompt, tool_schemas):
        return decision

    async def fake_summarizer(
        prompt,
        received_decision,
        tool_result,
    ):
        assert received_decision is decision
        assert tool_result == 42

        clock["now"] += 0.25

        return "结果是 42。"

    async def run_test():
        registry = ToolRegistry()
        registry.register(ADD_NUMBERS_TOOL)

        store = InMemoryTraceStore()

        runner = AgentRunner(
            registry=registry,
            decision_maker=fake_decision_maker,
            summarizer=fake_summarizer,
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

    assert result.result == 42
    assert result.answer == "结果是 42。"

    assert trace is not None
    assert trace.request_id == "request-runner-123"
    assert trace.trace_id == "trace-runner-123"
    assert trace.status == "ok"

    assert [span.name for span in trace.spans] == [
        "agent.run",
        "model.decision",
        "tool.execute",
        "model.summary",
    ]

    root_span, model_span, tool_span, summary_span = trace.spans

    assert root_span.parent_span_id is None
    assert len({span.span_id for span in trace.spans}) == 4

    for span in trace.spans:
        assert span.status == "ok"
        assert span.error_type is None
        assert span.duration_ms >= 0

    for span in trace.spans[1:]:
        assert span.parent_span_id == root_span.span_id

    assert model_span.attributes == {
        "available_tool_count": 1,
    }
    assert tool_span.attributes == {
        "tool_name": "add_numbers",
    }

    assert summary_span.duration_ms == 250
    assert root_span.duration_ms == 250
    assert trace.duration_ms == 250


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
            summarizer=unexpected_summary,
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
        return make_tool_decision(
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
            summarizer=unexpected_summary,
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
        return make_tool_decision(
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
            summarizer=successful_summary,
        )

        return await runner.run(
            "请计算 17 + 25",
            request_id="request-save-failed",
            trace_id="trace-save-failed",
        )

    result = asyncio.run(run_test())

    assert result.result == 42
    assert result.answer == "结果是 42。"
    assert store.save_calls == 1


@pytest.mark.parametrize(
    "failure_stage",
    ["model", "tool", "summary"],
)
def test_runner_preserves_business_error_when_trace_save_fails(
    monkeypatch,
    failure_stage,
):
    failure = RuntimeError("原来的业务异常")
    store = FailingTraceStore()
    summary_calls = []

    async def fake_decision_maker(prompt, tool_schemas):
        if failure_stage == "model":
            raise failure

        return make_tool_decision(
            name="add_numbers",
            arguments={"a": 17, "b": 25},
        )

    def fake_execute(name, arguments):
        raise failure

    async def fake_summarizer(prompt, decision, tool_result):
        summary_calls.append(tool_result)
        assert failure_stage == "summary"
        assert tool_result == 42
        raise failure

    async def run_test():
        registry = ToolRegistry()
        registry.register(ADD_NUMBERS_TOOL)

        if failure_stage == "tool":
            monkeypatch.setattr(
                registry,
                "execute",
                fake_execute,
            )

        runner = AgentRunner(
            registry=registry,
            decision_maker=fake_decision_maker,
            summarizer=fake_summarizer,
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
    assert summary_calls == (
        [42] if failure_stage == "summary" else []
    )


def test_runner_logs_trace_save_failure(caplog):
    store = FailingTraceStore()

    async def fake_decision_maker(prompt, tool_schemas):
        return make_tool_decision(
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
            summarizer=successful_summary,
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

    assert result.result == 42
    assert result.answer == "结果是 42。"
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


def test_runner_records_summary_failure():
    failure = OllamaSummaryError("模拟总结失败")

    tool_call = ToolCall(
        name="add_numbers",
        arguments={"a": 17, "b": 25},
    )

    decision = ToolDecision(
        tool_call=tool_call,
        assistant_message={
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": tool_call.model_dump()},
            ],
        },
    )

    async def fake_decision_maker(prompt, tool_schemas):
        return decision

    async def fake_summarizer(
        prompt,
        received_decision,
        tool_result,
    ):
        assert tool_result == 42
        raise failure

    async def run_test():
        registry = ToolRegistry()
        registry.register(ADD_NUMBERS_TOOL)

        store = InMemoryTraceStore()

        runner = AgentRunner(
            registry=registry,
            decision_maker=fake_decision_maker,
            summarizer=fake_summarizer,
            trace_store=store,
        )

        with pytest.raises(OllamaSummaryError) as exc_info:
            await runner.run(
                "请计算 17 + 25",
                request_id="request-summary-failed",
                trace_id="trace-summary-failed",
            )

        assert exc_info.value is failure

        return await store.get("trace-summary-failed")

    trace = asyncio.run(run_test())

    assert trace is not None
    assert trace.request_id == "request-summary-failed"
    assert trace.status == "error"

    assert [span.name for span in trace.spans] == [
        "agent.run",
        "model.decision",
        "tool.execute",
        "model.summary",
    ]

    assert [span.status for span in trace.spans] == [
        "error",
        "ok",
        "ok",
        "error",
    ]

    assert [span.error_type for span in trace.spans] == [
        "OllamaSummaryError",
        None,
        None,
        "OllamaSummaryError",
    ]

    root_span = trace.spans[0]
    assert root_span.parent_span_id is None

    for span in trace.spans[1:]:
        assert span.parent_span_id == root_span.span_id
        assert span.duration_ms >= 0
