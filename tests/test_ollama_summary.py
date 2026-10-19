import asyncio
import json
from copy import deepcopy

import httpx

from app.services import ollama_service


def test_build_summary_payload_preserves_assistant_message():
    assistant_message = {
        "role": "assistant",
        "content": "",
        "thinking": "test-context",
        "tool_calls": [
            {
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

    expected_assistant = deepcopy(
        assistant_message
    )

    payload = ollama_service.build_summary_payload(
        model="test-model",
        prompt="请计算 17 + 25",
        assistant_message=assistant_message,
        tool_name="add_numbers",
        tool_result=42,
    )

    assert payload == {
        "model": "test-model",
        "messages": [
            {
                "role": "user",
                "content": "请计算 17 + 25",
            },
            expected_assistant,
            {
                "role": "tool",
                "tool_name": "add_numbers",
                "content": "42",
            },
        ],
        "stream": False,
    }

    assert assistant_message == expected_assistant


def test_chat_for_summary_sends_history_and_returns_data():
    assistant_message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "name": "add_numbers",
                    "arguments": {"a": 17, "b": 25},
                },
            },
        ],
    }

    expected_response = {
        "message": {
            "role": "assistant",
            "content": "17 加 25 的结果是 42。",
        },
        "done": True,
    }

    captured_request = {}

    def handle_request(
        request: httpx.Request,
    ) -> httpx.Response:
        captured_request["method"] = request.method
        captured_request["url"] = str(request.url)
        captured_request["payload"] = json.loads(
            request.content
        )

        return httpx.Response(
            status_code=200,
            json=expected_response,
        )

    async def run_test() -> dict:
        transport = httpx.MockTransport(
            handle_request
        )

        async with httpx.AsyncClient(
            transport=transport,
        ) as client:
            return await ollama_service.chat_for_summary(
                model="test-model",
                prompt="请计算 17 + 25",
                assistant_message=assistant_message,
                tool_name="add_numbers",
                tool_result=42,
                chat_url="http://ollama.test/api/chat",
                timeout_seconds=12.5,
                client=client,
            )

    data = asyncio.run(run_test())

    assert captured_request["method"] == "POST"
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
    assert data == expected_response
