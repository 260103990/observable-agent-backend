import asyncio

from app.agent.runner import AgentRunner
from app.schemas.agent import ToolCall, ToolDecision
from app.tools.calculator import ADD_NUMBERS_TOOL
from app.tools.registry import ToolRegistry


def test_tool_call_parses_model_decision():
    decision = {
        "name": "add_numbers",
        "arguments": {
            "a": 2,
            "b": 3,
        },
    }

    tool_call = ToolCall.model_validate(decision)

    assert tool_call.name == "add_numbers"
    assert tool_call.arguments == {
        "a": 2,
        "b": 3,
    }


def test_agent_runner_executes_selected_tool():
    async def fake_decision_maker(prompt, tool_schemas):
        assert prompt == "请计算 2 + 3"
        assert tool_schemas == [
            ADD_NUMBERS_TOOL.to_schema(),
        ]

        tool_call = ToolCall(
            name="add_numbers",
            arguments={"a": 2, "b": 3},
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

    async def fake_summarizer(prompt, decision, tool_result):
        assert tool_result == 5
        return "结果是 5。"

    registry = ToolRegistry()
    registry.register(ADD_NUMBERS_TOOL)

    runner = AgentRunner(
        registry=registry,
        decision_maker=fake_decision_maker,
        summarizer=fake_summarizer,
    )

    result = asyncio.run(
        runner.run("请计算 2 + 3")
    )

    assert result.result == 5
    assert result.answer == "结果是 5。"


def test_agent_runner_summarizes_tool_result():
    calls = []

    tool_call = ToolCall(
        name="add_numbers",
        arguments={"a": 17, "b": 25},
    )

    decision = ToolDecision(
        tool_call=tool_call,
        assistant_message={
            "role": "assistant",
            "content": "",
            "thinking": "test-context",
            "tool_calls": [
                {"function": tool_call.model_dump()},
            ],
        },
    )

    async def fake_decision_maker(
        prompt: str,
        tool_schemas: list[dict[str, object]],
    ) -> ToolDecision:
        calls.append("decision")

        assert prompt == "请计算 17 + 25"
        assert tool_schemas == [
            ADD_NUMBERS_TOOL.to_schema(),
        ]

        return decision

    async def fake_summarizer(
        prompt: str,
        received_decision: ToolDecision,
        tool_result: object,
    ) -> str:
        calls.append("summary")

        assert prompt == "请计算 17 + 25"
        assert received_decision is decision
        assert tool_result == 42

        return "17 加 25 的结果是 42。"

    registry = ToolRegistry()
    registry.register(ADD_NUMBERS_TOOL)

    runner = AgentRunner(
        registry=registry,
        decision_maker=fake_decision_maker,
        summarizer=fake_summarizer,
    )

    result = asyncio.run(
        runner.run("请计算 17 + 25")
    )

    assert result.result == 42
    assert result.answer == "17 加 25 的结果是 42。"
    assert calls == ["decision", "summary"]
