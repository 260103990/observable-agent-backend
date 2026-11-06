"""Public paper import and grounded-question contracts."""
from pydantic import BaseModel, Field, field_validator

from app.rag.answer_models import PaperAnswer


class PaperImportResponse(BaseModel):
    paper_id: str
    title: str
    page_count: int
    chunk_count: int
    trace_id: str


class PaperQuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=100, strict=True)

    @field_validator("question", mode="before")
    @classmethod
    def strip_question(cls, value):
        if isinstance(value, str):
            return value.strip()
        return value


class PaperQuestionResponse(PaperAnswer):
    trace_id: str
