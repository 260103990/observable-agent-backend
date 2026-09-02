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