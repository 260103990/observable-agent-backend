import asyncio

from app.agent.runner import AgentRunner
from app.schemas.agent import ToolCall
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
    async def fake_decision_maker(
        prompt: str,
        tool_schemas: list[dict[str, object]],
    ) -> ToolCall:
        assert prompt == "请计算 2 + 3"
        assert tool_schemas == [
            ADD_NUMBERS_TOOL.to_schema(),
        ]

        return ToolCall(
            name="add_numbers",
            arguments={
                "a": 2,
                "b": 3,
            },
        )

    registry = ToolRegistry()
    registry.register(ADD_NUMBERS_TOOL)

    runner = AgentRunner(
        registry=registry,
        decision_maker=fake_decision_maker,
    )

    result = asyncio.run(
        runner.run("请计算 2 + 3")
    )

    assert result == 5
