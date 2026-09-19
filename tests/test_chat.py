import pytest
from fastapi.testclient import TestClient
from app.services.ollama_service import (OllamaTimeoutError, OllamaConnectionError,OllamaResponseError,)
from app.main import app
from app.config import Settings, get_settings


client = TestClient(app)


def test_chat_returns_llm_answer(monkeypatch):
    async def fake_generate(
        model: str,
        prompt: str,
        generate_url: str,
        timeout_seconds: float,
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

    async def fake_generate(model: str, prompt: str, generate_url: str, timeout_seconds: float) -> str:
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

    async def fake_generate(model: str, prompt: str, generate_url: str, timeout_seconds: float) -> str:
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
    async def fake_generate(model: str, prompt: str, generate_url: str, timeout_seconds: float) -> str:
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
    async def fake_generate(model: str, prompt: str, generate_url: str, timeout_seconds: float) -> str:
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

def test_chat_uses_configured_model(monkeypatch):
    received_models = []

    async def fake_generate(model: str, prompt: str, generate_url: str, timeout_seconds: float) -> str:
        received_models.append(model)
        return "模型正常"

    monkeypatch.setattr(
        "app.main.generate",
        fake_generate,
    )

    test_settings = Settings(
        ollama_model="qwen3:4b",
        _env_file=None,
    )

    app.dependency_overrides[get_settings] = (
        lambda: test_settings
    )

    try:
        response = client.post(
            "/chat",
            json={"message": "Hello"},
        )
    finally:
        app.dependency_overrides.pop(
            get_settings,
            None,
        )

    assert response.status_code == 200
    assert received_models == ["qwen3:4b"]

def test_chat_passes_ollama_settings_to_generate(monkeypatch):
    received_settings = {}

    async def fake_generate(
        model: str,
        prompt: str,
        generate_url: str | None = None,
        timeout_seconds: float | None = None,
    ) -> str:
        received_settings["generate_url"] = generate_url
        received_settings["timeout_seconds"] = timeout_seconds
        return "模型正常"

    monkeypatch.setattr(
        "app.main.generate",
        fake_generate,
    )

    test_settings = Settings(
        ollama_generate_url=(
            "http://configured:11434/api/generate"
        ),
        ollama_timeout_seconds=30.0,
        _env_file=None,
    )

    app.dependency_overrides[get_settings] = (
        lambda: test_settings
    )

    try:
        response = client.post(
            "/chat",
            json={"message": "Hello"},
        )
    finally:
        app.dependency_overrides.pop(
            get_settings,
            None,
        )

    assert response.status_code == 200
    assert received_settings == {
        "generate_url": (
            "http://configured:11434/api/generate"
        ),
        "timeout_seconds": 30.0,
    }

def test_chat_returns_502_when_ollama_returns_error(
    monkeypatch,
):
    async def fake_generate(
        model: str,
        prompt: str,
        generate_url: str,
        timeout_seconds: float,
    ) -> str:
        raise OllamaResponseError(
            "Ollama returned an error response"
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

    assert response.status_code == 502
    assert response.json() == {
        "detail": "Ollama returned an error response",
    }