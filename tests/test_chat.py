import pytest
from fastapi.testclient import TestClient
from app.services.ollama_service import (OllamaTimeoutError, OllamaConnectionError,)
from app.main import app


client = TestClient(app)


def test_chat_returns_llm_answer(monkeypatch):
    async def fake_generate(
        model: str,
        prompt: str,
    ) -> str:
        assert model == "qwen2.5:1.5b"
        assert prompt == "Hello"
        return "模型正常"

    monkeypatch.setattr(
        "app.main.generate",
        fake_generate,
        raising=False,
    )

    response = client.post(
        "/chat",
        json={"message": "Hello"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "模型正常",
    }
@pytest.mark.parametrize("message", ["", "   "])
def test_chat_rejects_blank_message(monkeypatch, message):
    calls = []

    async def fake_generate(model: str, prompt: str) -> str:
        calls.append(prompt)
        return "不应该调用模型"

    monkeypatch.setattr(
        "app.main.generate",
        fake_generate,
    )

    response = client.post(
        "/chat",
        json={"message": message},
    )

    assert response.status_code == 422
    assert calls == []

def test_chat_strips_surrounding_whitespace(monkeypatch):
    received_prompts = []

    async def fake_generate(model: str, prompt: str) -> str:
        received_prompts.append(prompt)
        return "模型正常"

    monkeypatch.setattr("app.main.generate", fake_generate)

    response = client.post(
        "/chat",
        json={"message": "  Hello world  "},
    )

    assert response.status_code == 200
    assert received_prompts == ["Hello world"]

    
def test_chat_returns_504_when_ollama_times_out(monkeypatch):
    async def fake_generate(model: str, prompt: str) -> str:
        raise OllamaTimeoutError(
            "Ollama request timed out"
        )

    monkeypatch.setattr(
        "app.main.generate",
        fake_generate,
    )

    error_client = TestClient(
        app,
        raise_server_exceptions=False,
    )

    response = error_client.post(
        "/chat",
        json={"message": "Hello"},
    )

    assert response.status_code == 504
    assert response.json() == {
        "detail": "Ollama request timed out",
    }

def test_chat_returns_503_when_ollama_is_unavailable(monkeypatch):
    async def fake_generate(model: str, prompt: str) -> str:
        raise OllamaConnectionError(
            "Ollama service is unavailable"
        )

    monkeypatch.setattr(
        "app.main.generate",
        fake_generate,
    )

    error_client = TestClient(
        app,
        raise_server_exceptions=False,
    )

    response = error_client.post(
        "/chat",
        json={"message": "Hello"},
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Ollama service is unavailable",
    }