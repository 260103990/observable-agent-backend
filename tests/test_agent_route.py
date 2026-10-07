import asyncio

import pytest
from pydantic import ValidationError


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
    from app.schemas.agent import ToolCall

    captured = {}

    class FakeDecisionMaker:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def __call__(
            self,
            prompt: str,
            tool_schemas: list[dict[str, object]],
        ) -> ToolCall:
            captured["prompt"] = prompt
            captured["tool_schemas"] = tool_schemas

            return ToolCall(
                name="add_numbers",
                arguments={
                    "a": 17,
                    "b": 25,
                },
            )

    monkeypatch.setattr(
        dependencies,
        "OllamaDecisionMaker",
        FakeDecisionMaker,
    )

    settings = Settings(
        ollama_model="test-model",
        ollama_chat_url=(
            "http://ollama.test/api/chat"
        ),
        ollama_timeout_seconds=12.5,
    )

    runner = dependencies.get_agent_runner(settings)

    result = asyncio.run(
        runner.run("请计算 17 + 25")
    )

    assert result == 42
    assert captured["model"] == "test-model"
    assert captured["chat_url"] == (
        "http://ollama.test/api/chat"
    )
    assert captured["timeout_seconds"] == 12.5
    assert captured["client"] is None
    assert captured["prompt"] == "请计算 17 + 25"
    assert captured["tool_schemas"][0]["name"] == (
        "add_numbers"
    )
