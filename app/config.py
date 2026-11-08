from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from functools import lru_cache

DEFAULT_EMBEDDING_QUERY_INSTRUCTION = (
    "Given a question about a scientific paper, retrieve relevant passages that answer the question."
)


class Settings(BaseSettings):
    readiness_timeout_seconds: float = Field(default=1.0, gt=0, le=10, allow_inf_nan=False)

    web_search_cache_coalesce_enabled: bool = True
    web_search_cache_max_inflight: int = Field(default=64, gt=0, le=1024)
    web_search_cache_max_waiters: int = Field(default=32, gt=0, le=256)
    web_search_cache_enabled: bool = False
    web_search_cache_ttl_seconds: float = Field(default=300.0, gt=0, le=86400, allow_inf_nan=False)
    web_search_cache_max_entries: int = Field(default=128, gt=0, le=10000)
    web_search_cache_max_bytes: int = Field(default=8*1024*1024, gt=0, le=128*1024*1024)
    web_search_cache_max_entry_bytes: int = Field(default=262144, gt=0, le=1048576)
    web_search_cache_timeout_seconds: float = Field(default=0.05, gt=0, le=1.0, allow_inf_nan=False)
    web_search_enabled: bool = False
    web_search_timeout_seconds: float = Field(default=30.0, gt=0, le=120, allow_inf_nan=False)
    web_search_max_response_bytes: int = Field(default=262144, gt=0, le=1048576)
    web_search_max_evidence_items: int = Field(default=10, gt=0, le=20)
    web_search_max_evidence_chars: int = Field(default=12000, gt=0, le=48000)

    # V2 loop limits are server-owned; callers cannot raise them per request.
    agent_loop_max_final_repairs: int = Field(default=1, ge=0, le=1)
    agent_loop_max_rounds: int = Field(default=5, ge=1, le=12)
    agent_loop_max_tool_calls: int = Field(default=8, ge=1, le=32)
    agent_loop_max_calls_per_turn: int = Field(default=4, ge=1, le=8)
    agent_loop_model_timeout_seconds: float = Field(default=120.0, gt=0, allow_inf_nan=False)
    agent_loop_tool_timeout_seconds: float = Field(default=60.0, gt=0, allow_inf_nan=False)
    agent_loop_run_timeout_seconds: float = Field(default=240.0, gt=0, allow_inf_nan=False)
    agent_loop_trace_timeout_seconds: float = Field(default=5.0, gt=0, le=30, allow_inf_nan=False)
    agent_loop_max_history_chars: int = Field(default=64000, gt=0, le=256000)
    agent_loop_max_tool_result_chars: int = Field(default=32000, gt=0, le=128000)

    rag_enabled: bool = False
    rag_retrieval_strategy: Literal['vector','hybrid'] = 'vector'
    rag_storage_dir: Path = Path("data/papers")
    rag_max_file_bytes: int = Field(default=20*1024*1024, gt=0)
    rag_max_pages: int = Field(default=100, gt=0)
    rag_max_text_chars: int = Field(default=1_000_000, gt=0)
    rag_max_chunks: int = Field(default=2000, gt=0)
    rag_max_concurrency: int = Field(default=2, gt=0)
    rag_queue_timeout_seconds: float = Field(default=5.0, gt=0, allow_inf_nan=False)
    rag_chunk_size_chars: int = Field(default=1200, gt=0)
    rag_overlap_chars: int = Field(default=150, ge=0)

    @model_validator(mode="after")
    def validate_rag_overlap(self):
        if self.rag_overlap_chars >= self.rag_chunk_size_chars:
            raise ValueError("rag_overlap_chars must be smaller than rag_chunk_size_chars")
        return self

    deepseek_api_key: SecretStr | None = None
    answer_scope_policy_enabled: bool = False
    deepseek_model: str = "deepseek-flash"
    deepseek_chat_url: str = "https://api.deepseek.com/chat/completions"
    deepseek_timeout_seconds: float = Field(default=120.0, gt=0, allow_inf_nan=False)
    deepseek_max_tokens: int = Field(default=2048, gt=0, le=393216)
    rag_max_evidence_chars: int = Field(default=12000, gt=0)
    rag_max_evidence_items: int = Field(default=10, gt=0)

    @field_validator("deepseek_model", "deepseek_chat_url")
    @classmethod
    def validate_answer_strings(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Answer configuration must not be blank")
        return value.strip()

    @field_validator("deepseek_api_key")
    @classmethod
    def optional_deepseek_key(cls, value: SecretStr | None) -> SecretStr | None:
        return value if value is not None and value.get_secret_value().strip() else None

    ollama_embedding_model: str = "qwen3-embedding:0.6b"
    ollama_embed_url: str = "http://127.0.0.1:11434/api/embed"
    ollama_embedding_dimensions: int = Field(default=1024, gt=0)
    ollama_embedding_batch_size: int = Field(default=16, gt=0)
    ollama_embedding_timeout_seconds: float = Field(default=120.0, gt=0, allow_inf_nan=False)
    embedding_query_instruction: str = DEFAULT_EMBEDDING_QUERY_INSTRUCTION

    @field_validator("ollama_embedding_model", "ollama_embed_url", "embedding_query_instruction")
    @classmethod
    def validate_embedding_strings(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Embedding configuration must not be blank")
        return value.strip()

    database_url: str | None = None
    ollama_model: str = "qwen2.5:1.5b"
    ollama_generate_url: str = (
        "http://127.0.0.1:11434/api/generate"
    )
    ollama_chat_url: str = (
        "http://127.0.0.1:11434/api/chat"
    )
    ollama_timeout_seconds: float = 120.0

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

@lru_cache
def get_settings() -> Settings:
    return Settings()
