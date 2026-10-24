import asyncio
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.dependencies import get_agent_runner
from app.main import app
from app.agent.ollama_summarizer import OllamaSummaryError
from app.schemas.agent import AgentRunResult, ToolCall, ToolDecision
from app.agent.ollama_decision_maker import (
    OllamaDecisionError,
)
from app.services.ollama_service import (
    OllamaConnectionError,
    OllamaResponseError,
    OllamaTimeoutError,
)
from app.tools.registry import (
    ToolArgumentsError,
    ToolNotFoundError,
)


@pytest.fixture(autouse=True)
def clear_dependency_overrides():
    yield

    app.dependency_overrides.clear()


class FakeRunner:
    def __init__(
        self,
        *,
        result: int = 42,
        answer: str = "结果是 42。",
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.answer = answer
        self.error = error
        self.prompts: list[str] = []
        self.request_ids: list[str | None] = []
        self.trace_ids: list[str | None] = []

    async def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        trace_id: str | None = None,
    ) -> AgentRunResult:
        self.prompts.append(prompt)
        self.request_ids.append(request_id)
        self.trace_ids.append(trace_id)

        if self.error is not None:
            raise self.error

        return AgentRunResult(
            result=self.result,
            answer=self.answer,
        )


def test_agent_run_request_strips_prompt():
    from app.schemas.agent import AgentRunRequest

    request = AgentRunRequest(
        prompt="  请计算 17 + 25  ",
    )

    assert request.prompt == "请计算 17 + 25"


@pytest.mark.parametrize("prompt", ["", "   "])
def test_agent_run_request_rejects_blank_prompt(prompt):
    from app.schemas.agent import AgentRunRequest

    with pytest.raises(ValidationError):
        AgentRunRequest(prompt=prompt)


def test_get_agent_runner_uses_settings_and_registered_tool(
    monkeypatch,
):
    from app.api import dependencies
    from app.config import Settings

    captured = {}

    class FakeDecisionMaker:
        def __init__(self, **kwargs):
            captured["decision_settings"] = kwargs

        async def __call__(self, prompt, tool_schemas):
            captured["prompt"] = prompt
            captured["tool_schemas"] = tool_schemas

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

            captured["decision"] = decision
            return decision

    class FakeSummarizer:
        def __init__(self, **kwargs):
            captured["summary_settings"] = kwargs

        async def __call__(self, prompt, decision, tool_result):
            captured["summary_args"] = (
                prompt,
                decision,
                tool_result,
            )
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

    settings = Settings(
        ollama_model="test-model",
        ollama_chat_url="http://ollama.test/api/chat",
        ollama_timeout_seconds=12.5,
    )

    runner = dependencies.get_agent_runner(
        settings=settings,
        trace_store=dependencies.get_trace_store(),
    )

    result = asyncio.run(
        runner.run(
            "请计算 17 + 25",
            request_id="request-settings-test",
            trace_id="trace-settings-test",
        )
    )

    expected_settings = {
        "model": "test-model",
        "chat_url": "http://ollama.test/api/chat",
        "timeout_seconds": 12.5,
        "client": None,
    }

    assert captured["decision_settings"] == expected_settings
    assert captured["summary_settings"] == expected_settings
    assert captured["prompt"] == "请计算 17 + 25"
    assert captured["tool_schemas"][0]["name"] == "add_numbers"

    summary_prompt, decision, tool_result = captured["summary_args"]

    assert summary_prompt == "请计算 17 + 25"
    assert decision is captured["decision"]
    assert tool_result == 42
    assert result.result == 42
    assert result.answer == "结果是 42。"


def test_agent_run_returns_tool_result():
    runner = FakeRunner(
        result=42,
        answer="17 加 25 的结果是 42。",
    )

    app.dependency_overrides[get_agent_runner] = lambda: runner

    client = TestClient(
        app,
        raise_server_exceptions=False,
    )

    response = client.post(
        "/agent/run",
        json={"prompt": "请计算 17 + 25"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "result": 42,
        "answer": "17 加 25 的结果是 42。",
    }
    assert runner.prompts == ["请计算 17 + 25"]


def test_agent_run_passes_request_id_to_runner():
    runner = FakeRunner(result=42)

    app.dependency_overrides[get_agent_runner] = (
        lambda: runner
    )

    client = TestClient(app)

    response = client.post(
        "/agent/run",
        headers={
            "X-Request-ID": "agent-request-123",
        },
        json={
            "prompt": "请计算 17 + 25",
        },
    )

    assert response.status_code == 200
    assert runner.request_ids == [
        "agent-request-123",
    ]


def test_agent_run_generates_new_trace_id_for_each_request():
    runner = FakeRunner(result=42)

    app.dependency_overrides[get_agent_runner] = (
        lambda: runner
    )

    client = TestClient(app)

    first_response = client.post(
        "/agent/run",
        json={
            "prompt": "第一次",
        },
    )
    second_response = client.post(
        "/agent/run",
        json={
            "prompt": "第二次",
        },
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200

    first_trace_id, second_trace_id = runner.trace_ids

    assert first_trace_id is not None
    assert second_trace_id is not None

    UUID(first_trace_id)
    UUID(second_trace_id)

    assert first_trace_id != second_trace_id


def test_agent_run_returns_trace_id_header():
    runner = FakeRunner(result=42)

    app.dependency_overrides[get_agent_runner] = (
        lambda: runner
    )

    client = TestClient(app)

    response = client.post(
        "/agent/run",
        json={
            "prompt": "请计算 17 + 25",
        },
    )

    assert response.status_code == 200

    trace_id = runner.trace_ids[0]

    assert trace_id is not None
    assert response.headers.get(
        "X-Trace-ID"
    ) == trace_id


def test_agent_run_strips_prompt_before_runner():
    runner = FakeRunner(result=42)

    app.dependency_overrides[get_agent_runner] = (
        lambda: runner
    )

    client = TestClient(app)

    response = client.post(
        "/agent/run",
        json={
            "prompt": "  请计算 17 + 25  ",
        },
    )

    assert response.status_code == 200
    assert runner.prompts == [
        "请计算 17 + 25",
    ]


@pytest.mark.parametrize("prompt", ["", "   "])
def test_agent_run_rejects_blank_prompt(prompt):
    runner = FakeRunner(result=42)

    app.dependency_overrides[get_agent_runner] = (
        lambda: runner
    )

    client = TestClient(app)

    response = client.post(
        "/agent/run",
        json={
            "prompt": prompt,
        },
    )

    assert response.status_code == 422
    assert runner.prompts == []


@pytest.mark.parametrize(
    ("error", "status_code", "detail"),
    [
        (
            OllamaTimeoutError("timeout"),
            504,
            "Ollama request timed out",
        ),
        (
            OllamaConnectionError("unavailable"),
            503,
            "Ollama service is unavailable",
        ),
        (
            OllamaResponseError("bad response"),
            502,
            "Ollama returned an error response",
        ),
        (
            OllamaDecisionError("invalid decision"),
            502,
            "Ollama returned an invalid tool decision",
        ),
        (
            ToolNotFoundError("unknown tool"),
            502,
            "Ollama selected an unknown tool",
        ),
        (
            OllamaSummaryError("invalid summary"),
            502,
            "Ollama returned an invalid summary",
        ),
        (
            ToolArgumentsError("invalid arguments"),
            502,
            "Ollama returned invalid tool arguments",
        ),
    ],
)
def test_agent_run_maps_known_errors(
    error,
    status_code,
    detail,
):
    runner = FakeRunner(error=error)

    app.dependency_overrides[get_agent_runner] = (
        lambda: runner
    )

    client = TestClient(
        app,
        raise_server_exceptions=False,
    )

    response = client.post(
        "/agent/run",
        json={
            "prompt": "请计算 17 + 25",
        },
    )

    assert response.status_code == status_code
    assert response.json() == {
        "detail": detail,
    }


def test_agent_route_has_agent_tag():
    client = TestClient(app)

    schema = client.get("/openapi.json").json()

    tags = schema["paths"]["/agent/run"]["post"].get(
        "tags",
        [],
    )

    assert "agent" in tags
