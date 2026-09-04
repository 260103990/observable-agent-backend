import pytest
from fastapi.testclient import TestClient

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