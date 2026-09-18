from pydantic_settings import BaseSettings, SettingsConfigDict

from functools import lru_cache

class Settings(BaseSettings):
    ollama_model: str = "qwen2.5:1.5b"
    ollama_generate_url: str = (
        "http://127.0.0.1:11434/api/generate"
    )
    ollama_timeout_seconds: float = 120.0

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

@lru_cache
def get_settings() -> Settings:
    return Settings()