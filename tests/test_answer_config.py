import pytest
from pydantic import SecretStr, ValidationError
from app.config import Settings


NAMES = ("DEEPSEEK_API_KEY", "DEEPSEEK_MODEL", "DEEPSEEK_CHAT_URL", "DEEPSEEK_TIMEOUT_SECONDS",
         "DEEPSEEK_MAX_TOKENS", "RAG_MAX_EVIDENCE_CHARS", "RAG_MAX_EVIDENCE_ITEMS")


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    for name in NAMES:
        monkeypatch.delenv(name, raising=False)


def test_answer_settings_default_key_optional_and_independent():
    settings = Settings(_env_file=None)
    assert settings.deepseek_api_key is None
    assert settings.deepseek_model == "deepseek-flash"
    assert settings.deepseek_chat_url == "https://api.deepseek.com/chat/completions"
    assert settings.deepseek_timeout_seconds == 120
    assert settings.deepseek_max_tokens == 2048
    assert settings.rag_max_evidence_chars == 12000
    assert settings.rag_max_evidence_items == 10
    assert settings.ollama_model == "qwen2.5:1.5b"


def test_answer_settings_read_env_and_mask_key(monkeypatch):
    for name, value in zip(NAMES, ("fake-private-key", "custom-model", "https://configured/chat/completions", "8", "512", "4000", "3")):
        monkeypatch.setenv(name, value)
    settings = Settings(_env_file=None)
    assert isinstance(settings.deepseek_api_key, SecretStr)
    assert settings.deepseek_api_key.get_secret_value() == "fake-private-key"
    assert "fake-private-key" not in repr(settings)
    assert settings.deepseek_model == "custom-model"
    assert settings.deepseek_timeout_seconds == 8
    assert settings.deepseek_max_tokens == 512
    assert settings.rag_max_evidence_chars == 4000
    assert settings.rag_max_evidence_items == 3


@pytest.mark.parametrize("name,value", [("DEEPSEEK_MODEL", " "), ("DEEPSEEK_CHAT_URL", " "),
    ("DEEPSEEK_TIMEOUT_SECONDS", "0"), ("DEEPSEEK_TIMEOUT_SECONDS", "nan"),
    ("DEEPSEEK_MAX_TOKENS", "0"), ("RAG_MAX_EVIDENCE_CHARS", "0"), ("RAG_MAX_EVIDENCE_ITEMS", "0")])
def test_invalid_answer_configuration_is_rejected(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
