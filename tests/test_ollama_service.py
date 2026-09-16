from importlib.util import find_spec

import pytest


def test_build_payload_uses_non_streaming_mode():
    if find_spec("app.services.ollama_service") is None:
        pytest.fail(
            "app/services/ollama_service.py 尚未实现"
        )

    from app.services.ollama_service import build_payload

    payload = build_payload(
        model="qwen2.5:1.5b",
        prompt="你好",
    )

    assert payload == {
        "model": "qwen2.5:1.5b",
        "prompt": "你好",
        "stream": False,
    }

import asyncio

import httpx

from app.services import ollama_service


def test_generate_sends_request_and_returns_answer():
    if not hasattr(ollama_service, "generate"):
        pytest.fail("generate() 尚未实现")

    captured_request = {}

    def handle_request(request: httpx.Request) -> httpx.Response:
        captured_request["method"] = request.method
        captured_request["url"] = str(request.url)

        return httpx.Response(
            status_code=200,
            json={
                "response": "模型正常",
                "done": True,
            },
        )

    async def run_test() -> str:
        transport = httpx.MockTransport(handle_request)

        async with httpx.AsyncClient(
            transport=transport,
        ) as client:
            return await ollama_service.generate(
                model="qwen2.5:1.5b",
                prompt="你好",
                client=client,
            )

    answer = asyncio.run(run_test())

    assert captured_request["method"] == "POST"
    assert captured_request["url"] == (
        "http://127.0.0.1:11434/api/generate"
    )
    assert answer == "模型正常"

    
def test_generate_translates_timeout():
    assert hasattr(
        ollama_service,
        "OllamaTimeoutError",
    ), "尚未定义 OllamaTimeoutError"

    def handle_request(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout(
            "Ollama 响应超时",
            request=request,
        )

    async def run_test() -> None:
        transport = httpx.MockTransport(handle_request)

        async with httpx.AsyncClient(
            transport=transport,
        ) as client:
            with pytest.raises(
                ollama_service.OllamaTimeoutError,
            ):
                await ollama_service.generate(
                    model="qwen2.5:1.5b",
                    prompt="你好",
                    client=client,
                )

    asyncio.run(run_test())


def test_generate_translates_connection_error():
    assert hasattr(
        ollama_service,
        "OllamaConnectionError",
    ), "尚未定义 OllamaConnectionError"

    def handle_request(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(
            "无法连接 Ollama",
            request=request,
        )

    async def run_test() -> None:
        transport = httpx.MockTransport(handle_request)

        async with httpx.AsyncClient(
            transport=transport,
        ) as client:
            with pytest.raises(
                ollama_service.OllamaConnectionError,
            ):
                await ollama_service.generate(
                    model="qwen2.5:1.5b",
                    prompt="你好",
                    client=client,
                )

    asyncio.run(run_test())