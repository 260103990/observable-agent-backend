import asyncio
import pytest
import httpx

from app.tools.calculator import ADD_NUMBERS_TOOL


def test_ollama_decision_maker_preserves_assistant_message():
    from app.agent.ollama_decision_maker import (
        OllamaDecisionMaker,
    )

    assistant_message = {
        "role": "assistant",
        "content": "",
        "thinking": "test-context",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "index": 0,
                    "name": "add_numbers",
                    "arguments": {
                        "a": 17,
                        "b": 25,
                    },
                },
            },
        ],
    }

    def handle_request(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={
                "message": assistant_message,
                "done": True,
            },
        )

    async def run_test():
        transport = httpx.MockTransport(handle_request)

        async with httpx.AsyncClient(
            transport=transport,
        ) as client:
            decision_maker = OllamaDecisionMaker(
                model="qwen2.5:1.5b",
                chat_url="http://configured:11434/api/chat",
                timeout_seconds=30.0,
                client=client,
            )

            return await decision_maker(
                "请计算 17 + 25",
                [ADD_NUMBERS_TOOL.to_schema()],
            )

    decision = asyncio.run(run_test())

    assert decision.tool_call.name == "add_numbers"
    assert decision.tool_call.arguments == {
        "a": 17,
        "b": 25,
    }
    assert decision.assistant_message == assistant_message


@pytest.mark.parametrize(
    "tool_calls",
    [
        [],
        [
            {
                "type": "function",
                "function": {
                    "name": "add_numbers",
                    "arguments": {
                        "a": 17,
                        "b": 25,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "add_numbers",
                    "arguments": {
                        "a": 1,
                        "b": 2,
                    },
                },
            },
        ],
    ],
)
def test_ollama_decision_maker_rejects_wrong_call_count(
    tool_calls,
):
    from app.agent.ollama_decision_maker import (
        OllamaDecisionError,
        OllamaDecisionMaker,
    )

    def handle_request(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": tool_calls,
                },
                "done": True,
            },
        )

    async def run_test() -> None:
        transport = httpx.MockTransport(
            handle_request
        )

        async with httpx.AsyncClient(
            transport=transport,
        ) as client:
            decision_maker = OllamaDecisionMaker(
                model="qwen2.5:1.5b",
                chat_url=(
                    "http://configured:11434/api/chat"
                ),
                timeout_seconds=30.0,
                client=client,
            )

            with pytest.raises(
                OllamaDecisionError,
                match="exactly one tool call",
            ):
                await decision_maker(
                    "请计算 17 + 25",
                    [
                        ADD_NUMBERS_TOOL.to_schema(),
                    ],
                )

    asyncio.run(run_test())


def test_ollama_decision_maker_rejects_malformed_response():
    from app.agent.ollama_decision_maker import (
        OllamaDecisionError,
        OllamaDecisionMaker,
    )

    def handle_request(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                },
                "done": True,
            },
        )

    async def run_test() -> None:
        transport = httpx.MockTransport(
            handle_request
        )

        async with httpx.AsyncClient(
            transport=transport,
        ) as client:
            decision_maker = OllamaDecisionMaker(
                model="qwen2.5:1.5b",
                chat_url=(
                    "http://configured:11434/api/chat"
                ),
                timeout_seconds=30.0,
                client=client,
            )

            with pytest.raises(
                OllamaDecisionError,
                match="invalid tool call",
            ):
                await decision_maker(
                    "请计算 17 + 25",
                    [
                        ADD_NUMBERS_TOOL.to_schema(),
                    ],
                )

    asyncio.run(run_test())


def test_agent_runner_executes_ollama_tool_decision():
    from app.agent.ollama_decision_maker import (
        OllamaDecisionMaker,
    )
    from app.agent.runner import AgentRunner
    from app.tools.registry import ToolRegistry

    def handle_request(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "add_numbers",
                                "arguments": {
                                    "a": 17,
                                    "b": 25,
                                },
                            },
                        },
                    ],
                },
                "done": True,
            },
        )

    async def fake_summarizer(
        prompt,
        decision,
        tool_result,
    ):
        assert prompt == "请计算 17 + 25"
        assert decision.tool_call.name == "add_numbers"
        assert decision.tool_call.arguments == {
            "a": 17,
            "b": 25,
        }
        assert decision.assistant_message["role"] == "assistant"
        assert tool_result == 42

        return "结果是 42。"

    async def run_test() -> object:
        transport = httpx.MockTransport(
            handle_request
        )

        async with httpx.AsyncClient(
            transport=transport,
        ) as client:
            decision_maker = OllamaDecisionMaker(
                model="qwen2.5:1.5b",
                chat_url=(
                    "http://configured:11434/api/chat"
                ),
                timeout_seconds=30.0,
                client=client,
            )

            registry = ToolRegistry()
            registry.register(ADD_NUMBERS_TOOL)

            runner = AgentRunner(
                registry=registry,
                decision_maker=decision_maker,
                summarizer=fake_summarizer,
            )

            return await runner.run(
                "请计算 17 + 25"
            )

    result = asyncio.run(run_test())

    assert result.result == 42
    assert result.answer == "结果是 42。"
