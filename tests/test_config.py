from importlib.util import find_spec

import pytest


def test_settings_use_defaults():
    if find_spec("app.config") is None:
        pytest.fail("app/config.py 尚未实现")

    from app.config import Settings

    settings = Settings(_env_file=None)

    assert settings.ollama_model == "qwen2.5:1.5b"
    assert (
        settings.ollama_generate_url
        == "http://127.0.0.1:11434/api/generate"
    )
    assert settings.ollama_timeout_seconds == 120.0

def test_settings_read_environment_variables(monkeypatch):
    monkeypatch.setenv(
        "OLLAMA_MODEL",
        "qwen3:4b",
    )
    monkeypatch.setenv(
        "OLLAMA_GENERATE_URL",
        "http://ollama:11434/api/generate",
    )
    monkeypatch.setenv(
        "OLLAMA_TIMEOUT_SECONDS",
        "30",
    )

    from app.config import Settings

    settings = Settings(_env_file=None)

    assert settings.ollama_model == "qwen3:4b"
    assert (
        settings.ollama_generate_url
        == "http://ollama:11434/api/generate"
    )
    assert settings.ollama_timeout_seconds == 30.0

def test_settings_reject_invalid_timeout(monkeypatch):
    monkeypatch.setenv(
        "OLLAMA_TIMEOUT_SECONDS",
        "不是数字",
    )

    from pydantic import ValidationError

    from app.config import Settings

    with pytest.raises(ValidationError):
        Settings(_env_file=None)

def test_get_settings_reuses_same_instance():
    from app import config

    assert hasattr(
        config,
        "get_settings",
    ), "get_settings() 尚未实现"

    config.get_settings.cache_clear()

    try:
        first_settings = config.get_settings()
        second_settings = config.get_settings()

        assert first_settings is second_settings
    finally:
        config.get_settings.cache_clear()