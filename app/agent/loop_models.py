"""Explicit decisions, limits and terminal results; no persistent conversation."""
from typing import Literal
import json

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.rag.answer_models import EvidenceItem
from app.tools.web_models import WebEvidenceItem


def load_json_object(content: str) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    def invalid_constant(value):
        raise ValueError('Non-finite JSON number')
    value = json.loads(content, object_pairs_hook=unique, parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise ValueError('Expected a JSON object')
    json.dumps(value, allow_nan=False)
    return value


class LoopLimits(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid')

    max_final_repairs: int = Field(default=1, ge=0, le=1)
    max_rounds: int = Field(default=5, ge=1, le=12)
    max_tool_calls: int = Field(default=8, ge=1, le=32)
    max_calls_per_turn: int = Field(default=4, ge=1, le=8)
    model_timeout_seconds: float = Field(default=120.0, gt=0, allow_inf_nan=False)
    tool_timeout_seconds: float = Field(default=60.0, gt=0, allow_inf_nan=False)
    run_timeout_seconds: float = Field(default=240.0, gt=0, allow_inf_nan=False)
    trace_timeout_seconds: float = Field(default=5.0, gt=0, le=30, allow_inf_nan=False)
    max_history_chars: int = Field(default=64000, gt=0)
    max_tool_result_chars: int = Field(default=32000, gt=0)


class LoopToolCall(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid')
    id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,128}$')
    name: str = Field(pattern=r'^[A-Za-z0-9_-]{1,64}$')
    arguments: dict[str, object]


class LoopTurn(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid')
    content: str | None = None
    calls: tuple[LoopToolCall, ...] = ()

    @model_validator(mode='after')
    def nonempty(self):
        if not self.calls and (self.content is None or not self.content.strip()):
            raise ValueError('A turn needs tool calls or nonblank content')
        return self


class LoopAnswer(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')
    answer: str
    citations: tuple[EvidenceItem, ...] = ()
    web_citations: tuple[WebEvidenceItem, ...] = ()
    insufficient_evidence: bool = False
    reason: str | None = None
    omitted_evidence_count: int = Field(default=0,ge=0)
    omitted_web_evidence_count: int = Field(default=0,ge=0)


class LoopResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')
    final_repairs: int = Field(default=0, ge=0, le=1)
    status: Literal['completed', 'limit_reached']
    answer: str | None = None
    stop_reason: Literal['max_rounds', 'max_tool_calls'] | None = None
    rounds: int
    tool_calls: int
    citations: tuple[EvidenceItem, ...] = ()
    web_citations: tuple[WebEvidenceItem, ...] = ()
    insufficient_evidence: bool = False
    reason: str | None = None
    omitted_evidence_count: int = Field(default=0,ge=0)
    omitted_web_evidence_count: int = Field(default=0,ge=0)
