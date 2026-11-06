import asyncio

import httpx
import pytest

from app.config import Settings


NAMES = ("OLLAMA_EMBEDDING_MODEL", "OLLAMA_EMBED_URL", "OLLAMA_EMBEDDING_DIMENSIONS",
         "OLLAMA_EMBEDDING_BATCH_SIZE", "OLLAMA_EMBEDDING_TIMEOUT_SECONDS", "EMBEDDING_QUERY_INSTRUCTION")


@pytest.fixture(autouse=True)
def clear_embedding_environment(monkeypatch):
    for name in NAMES:
        monkeypatch.delenv(name, raising=False)


def test_embedding_defaults_do_not_replace_chat_model_settings():
    settings = Settings(_env_file=None)
    assert settings.ollama_embedding_model == "qwen3-embedding:0.6b"
    assert settings.ollama_embed_url == "http://127.0.0.1:11434/api/embed"
    assert settings.ollama_embedding_dimensions == 1024
    assert settings.ollama_embedding_batch_size == 16
    assert settings.ollama_embedding_timeout_seconds == 120.0
    assert settings.embedding_query_instruction.strip()
    assert settings.ollama_model == "qwen2.5:1.5b"


def test_embedding_settings_read_environment_and_build_client(monkeypatch):
    values = ("custom-embedding:v1", "http://configured:11434/api/embed", "3", "2", "7.5", "Custom instruction.")
    for name, value in zip(NAMES, values):
        monkeypatch.setenv(name, value)
    settings = Settings(_env_file=None)

    async def run():
        from app.rag.embedding import OllamaEmbeddingClient
        import json

        def handler(request):
            assert str(request.url) == values[1]
            payload = json.loads(request.content)
            assert payload["model"] == values[0]
            assert payload["dimensions"] == 3
            assert payload["input"] == ["Instruct: Custom instruction.\nQuery:question"]
            assert request.extensions["timeout"]["read"] == 7.5
            return httpx.Response(200, json={"model": values[0], "embeddings": [[1, 2, 3]]})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            embedding = OllamaEmbeddingClient.from_settings(settings, client=client)
            assert await embedding.embed_query("question") == [1.0, 2.0, 3.0]
    asyncio.run(run())


@pytest.mark.parametrize("name,value", [
    ("OLLAMA_EMBEDDING_DIMENSIONS", "0"), ("OLLAMA_EMBEDDING_BATCH_SIZE", "0"),
    ("OLLAMA_EMBEDDING_TIMEOUT_SECONDS", "0"), ("OLLAMA_EMBEDDING_TIMEOUT_SECONDS", "nan"),
    ("OLLAMA_EMBEDDING_MODEL", " "), ("EMBEDDING_QUERY_INSTRUCTION", " "),
])
def test_invalid_embedding_environment_fails_settings_validation(monkeypatch, name, value):
    from pydantic import ValidationError

    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
