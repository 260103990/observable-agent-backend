"""Persistence contracts; separate from the PDF reader and transient mapping."""
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.rag.models import TextChunk


class EmbeddingSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    dimensions: Literal[1024] = 1024
    query_instruction: str = Field(min_length=1)
    representation_version: str = "qwen-instruct-v1"

    @field_validator("model", "revision", "query_instruction", "representation_version")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Embedding identity must not be blank")
        return value


class PaperInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    paper_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    embedding: EmbeddingSpec
    cleaner_version: str = "conservative-v1"
    splitter_version: str = "page-boundaries-v1"
    chunk_size_chars: int = Field(default=1200, gt=0)
    overlap_chars: int = Field(default=150, ge=0)

    @model_validator(mode="after")
    def check_overlap(self) -> Self:
        if self.overlap_chars >= self.chunk_size_chars:
            raise ValueError("overlap_chars must be smaller than chunk_size_chars")
        return self


class StoredPage(BaseModel):
    page_number: int
    raw_text: str
    text: str


class StoredPaper(BaseModel):
    paper: PaperInfo
    pages: list[StoredPage]
    chunks: list[TextChunk]


class RetrievedChunk(BaseModel):
    paper_id: str
    chunk: TextChunk
    score: float
