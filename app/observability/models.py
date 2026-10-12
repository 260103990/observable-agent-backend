from pydantic import BaseModel


class SpanRecord(BaseModel):
    span_id: str
    parent_span_id: str | None
    name: str
    status: str
    duration_ms: float
    attributes: dict[str, str | int | float | bool]
    error_type: str | None


class TraceRecord(BaseModel):
    trace_id: str
    request_id: str
    status: str
    duration_ms: float
    spans: list[SpanRecord]
