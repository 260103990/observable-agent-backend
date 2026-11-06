"""Structured model output and immutable backend-owned citation snapshots."""
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EvidenceItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_id: str = Field(pattern=r"^E[1-9][0-9]*$")
    paper_id: str = Field(min_length=1)
    page_number: int = Field(gt=0)
    chunk_index: int = Field(ge=0)
    text: str
    raw_text: str
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    raw_start_char: int = Field(ge=0)
    raw_end_char: int = Field(gt=0)

    @model_validator(mode="after")
    def check_ranges(self) -> Self:
        if (not self.text.strip() or len(self.text) != self.end_char - self.start_char
                or not self.raw_text or len(self.raw_text) != self.raw_end_char - self.raw_start_char):
            raise ValueError("Evidence text must match its source range lengths")
        return self


class EvidenceBundle(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    paper_id: str
    items: tuple[EvidenceItem, ...]
    omitted_count: int = Field(default=0, ge=0)


class DraftClaim(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    text: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Claim must not be blank")
        return value.strip()

    @field_validator("evidence_ids")
    @classmethod
    def unique_ids(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("Claim evidence IDs must be unique")
        return value


class AnswerDraft(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    claims: list[DraftClaim] = Field(max_length=20)
    insufficient_evidence: bool
    reason: str | None

    @model_validator(mode="after")
    def coherent_status(self) -> Self:
        if self.insufficient_evidence:
            if self.reason is None or not self.reason.strip():
                raise ValueError("Insufficiency requires an explanation")
        elif not self.claims or self.reason is not None:
            raise ValueError("An answered result needs claims and no insufficiency reason")
        return self


class VerifiedClaim(BaseModel):
    model_config = ConfigDict(frozen=True)
    text: str
    evidence_ids: tuple[str, ...]


class PaperAnswer(BaseModel):
    model_config = ConfigDict(frozen=True)

    paper_id: str
    answer: str
    claims: tuple[VerifiedClaim, ...]
    citations: tuple[EvidenceItem, ...]
    insufficient_evidence: bool
    reason: str | None
    omitted_evidence_count: int = 0
    generation_model: str | None = None
    provider_request_id: str | None = None
