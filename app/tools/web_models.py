"""Search summaries are provider results, separate from PDF and fetched page text."""
from datetime import datetime
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SearchInput(BaseModel):
    model_config=ConfigDict(strict=True,extra='forbid')
    query: str=Field(min_length=1,max_length=128)
    count: int=Field(default=5,ge=1,le=10)

    @field_validator('query')
    @classmethod
    def nonblank(cls,value):
        if not value.strip():
            raise ValueError('Search query must not be blank')
        return value.strip()


class SearchResult(BaseModel):
    model_config=ConfigDict(frozen=True,strict=True,extra='forbid')
    source_type: Literal['web_search_result']='web_search_result'
    provider: Literal['deepseek']='deepseek'
    rank: int=Field(ge=1,le=10)
    title: str
    url: str=Field(min_length=1,max_length=8192)
    snippet: str
    summary: str
    published_at: str | None=None
    rerank_score: float=Field(ge=0,le=3,allow_inf_nan=False)
    authority: int=Field(ge=-1,le=4)

    @field_validator('url')
    @classmethod
    def safe_http_url(cls,value):
        if any(ch.isspace() or ord(ch)<32 or ord(ch)==127 for ch in value):
            raise ValueError('Invalid search URL')
        parsed=urlsplit(value)
        if (parsed.scheme.lower() not in ('http','https') or not parsed.hostname
                or parsed.username is not None or parsed.password is not None):
            raise ValueError('Search result needs a credential-free HTTP URL')
        parsed.port  # Reject malformed port values without making a request.
        return value

    @field_validator('published_at')
    @classmethod
    def aware_publication_time(cls,value):
        if value is not None and datetime.fromisoformat(value.replace('Z','+00:00')).tzinfo is None:
            raise ValueError('Publication timestamp requires a time zone')
        return value


class SearchBatch(BaseModel):
    model_config=ConfigDict(frozen=True,strict=True,extra='forbid')
    query: str
    retrieved_at: str
    trace_id: str
    response_body_bytes: int=Field(ge=0)
    results: tuple[SearchResult,...]


class WebEvidenceItem(SearchResult):
    evidence_id: str=Field(pattern=r'^W[1-9][0-9]*$')
    query: str
    retrieved_at: str
    search_trace_id: str
