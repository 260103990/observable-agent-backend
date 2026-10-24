import asyncio
import json

import httpx
import pytest

from app.agent import ollama_summarizer
from app.schemas import agent as agent_schemas
from app.services import ollama_service


def test_parse_summary_answer_returns_content():
    from app.agent.ollama_summarizer import parse_summary_answer

    data = {
        "message": {
            "role": "assistant",
            "content": "17 加 25 的结果是 42。",
        },
        "done": True,
    }

    answer = parse_summary_answer(data)

    assert answer == "17 加 25 的结果是 42。"


@pytest.mark.parametrize(
    "content",
    ["", "   ", "\t\n"],
)
def test_parse_summary_answer_rejects_blank_content(content):
    data = {
        "message": {
            "role": "assistant",
            "content": content,
        },
        "done": True,
    }

    with pytest.raises(
        ollama_summarizer.OllamaSummaryError,
    ):
        ollama_summarizer.parse_summary_answer(data)


@pytest.mark.parametrize(
    "data",
    [
        [],
        {},
        {"message": None},
        {"message": []},
        {"message": {}},
        {"message": {"content": None}},
        {"message": {"content": 42}},
        {"message": {"content": False}},
        {"message": {"content": []}},
        {"message": {"content": {}}},
    ],
    ids=[
        "root-not-object",
        "missing-message",
        "null-message",
        "message-not-object",
        "missing-content",
        "null-content",
        "number-content",
        "boolean-content",
        "array-content",
        "object-content",
    ],
)
def test_parse_summary_answer_rejects_invalid_response(data):
    with pytest.raises(
        ollama_summarizer.OllamaSummaryError,
    ):
        ollama_summarizer.parse_summary_answer(data)


def test_parse_summary_answer_accepts_empty_tool_calls():
    data = {
        "message": {
            "role": "assistant",
            "content": "结果是 42。",
            "tool_calls": [],
        },
        "done": True,
    }

    answer = ollama_summarizer.parse_summary_answer(data)

    assert answer == "结果是 42。"


def test_parse_summary_answer_rejects_tool_calls():
    data = {
        "message": {
            "role": "assistant",
            "content": "我还需要调用工具。",
            "tool_calls": [
                {
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
    }

    with pytest.raises(
        ollama_summarizer.OllamaSummaryError,
    ):
        ollama_summarizer.parse_summary_answer(data)


def test_ollama_summarizer_returns_final_answer():
    tool_call = agent_schemas.ToolCall(
        name="add_numbers",
        arguments={"a": 17, "b": 25},
    )

    assistant_message = {
        "role": "assistant",
        "content": "",
        "thinking": "test-context",
        "tool_calls": [
            {"function": tool_call.model_dump()},
        ],
    }

    decision = agent_schemas.ToolDecision(
        tool_call=tool_call,
        assistant_message=assistant_message,
    )

    captured_request = {}

    def handle_request(
        request: httpx.Request,
    ) -> httpx.Response:
        captured_request["url"] = str(request.url)
        captured_request["payload"] = json.loads(
            request.content
        )

        return httpx.Response(
            status_code=200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "结果是 42。",
                },
                "done": True,
            },
        )

    async def run_test() -> str:
        transport = httpx.MockTransport(
            handle_request
        )

        async with httpx.AsyncClient(
            transport=transport,
        ) as client:
            summarizer = ollama_summarizer.OllamaSummarizer(
                model="test-model",
                chat_url="http://ollama.test/api/chat",
                timeout_seconds=12.5,
                client=client,
            )

            return await summarizer(
                "请计算 17 + 25",
                decision,
                42,
            )

    answer = asyncio.run(run_test())

    assert answer == "结果是 42。"
    assert captured_request["url"] == (
        "http://ollama.test/api/chat"
    )
    assert captured_request["payload"] == (
        ollama_service.build_summary_payload(
            model="test-model",
            prompt="请计算 17 + 25",
            assistant_message=assistant_message,
            tool_name="add_numbers",
            tool_result=42,
        )
    )


def test_ollama_summarizer_rejects_invalid_json():
    tool_call = agent_schemas.ToolCall(
        name="add_numbers",
        arguments={"a": 17, "b": 25},
    )

    decision = agent_schemas.ToolDecision(
        tool_call=tool_call,
        assistant_message={
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": tool_call.model_dump()},
            ],
        },
    )

    def handle_request(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            content=b"not-json",
            headers={"Content-Type": "application/json"},
        )

    async def run_test() -> str:
        transport = httpx.MockTransport(handle_request)

        async with httpx.AsyncClient(
            transport=transport,
        ) as client:
            summarizer = ollama_summarizer.OllamaSummarizer(
                model="test-model",
                chat_url="http://ollama.test/api/chat",
                timeout_seconds=12.5,
                client=client,
            )

            return await summarizer(
                "请计算 17 + 25",
                decision,
                42,
            )

    with pytest.raises(
        ollama_summarizer.OllamaSummaryError,
    ):
        asyncio.run(run_test())


@pytest.mark.parametrize(
    ("failure", "expected_error"),
    [
        ("timeout", ollama_service.OllamaTimeoutError),
        ("connection", ollama_service.OllamaConnectionError),
        ("http_error", ollama_service.OllamaResponseError),
    ],
)
def test_ollama_summarizer_preserves_service_errors(
    failure,
    expected_error,
):
    tool_call = agent_schemas.ToolCall(
        name="add_numbers",
        arguments={"a": 17, "b": 25},
    )

    decision = agent_schemas.ToolDecision(
        tool_call=tool_call,
        assistant_message={
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": tool_call.model_dump()},
            ],
        },
    )

    def handle_request(
        request: httpx.Request,
    ) -> httpx.Response:
        if failure == "timeout":
            raise httpx.ReadTimeout(
                "test timeout",
                request=request,
            )

        if failure == "connection":
            raise httpx.ConnectError(
                "test connection failure",
                request=request,
            )

        return httpx.Response(
            status_code=500,
            text="upstream failed",
        )

    async def run_test() -> str:
        transport = httpx.MockTransport(handle_request)

        async with httpx.AsyncClient(
            transport=transport,
        ) as client:
            summarizer = ollama_summarizer.OllamaSummarizer(
                model="test-model",
                chat_url="http://ollama.test/api/chat",
                timeout_seconds=12.5,
                client=client,
            )

            return await summarizer(
                "请计算 17 + 25",
                decision,
                42,
            )

    with pytest.raises(expected_error):
        asyncio.run(run_test())
