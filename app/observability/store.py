from app.observability.models import SpanRecord, TraceRecord
from typing import Protocol
from asyncio import Lock

import asyncpg
import json


class TraceStore(Protocol):
    async def save(self, trace: TraceRecord) -> None:
        ...

    async def get(
        self,
        trace_id: str,
    ) -> TraceRecord | None:
        ...


class InMemoryTraceStore:
    def __init__(self) -> None:
        self._traces: dict[str, TraceRecord] = {}
        self._lock = Lock()

    async def save(self, trace: TraceRecord) -> None:
        async with self._lock:
            self._traces[trace.trace_id] = (
                trace.model_copy(deep=True)
            )

    async def get(
        self,
        trace_id: str,
    ) -> TraceRecord | None:
        async with self._lock:
            trace = self._traces.get(trace_id)

            if trace is None:
                return None

            return trace.model_copy(deep=True)


class PostgresTraceStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def save(self, trace: TraceRecord) -> None:
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    """
                    INSERT INTO traces (
                        trace_id,
                        request_id,
                        status,
                        duration_ms
                    )
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (trace_id)
                    DO UPDATE SET
                        request_id = EXCLUDED.request_id,
                        status = EXCLUDED.status,
                        duration_ms = EXCLUDED.duration_ms
                    """,
                    trace.trace_id,
                    trace.request_id,
                    trace.status,
                    trace.duration_ms,
                )

                await connection.execute(
                    "DELETE FROM spans WHERE trace_id = $1",
                    trace.trace_id,
                )

                span_rows = [
                    (span_index, span.span_id, span.parent_span_id, span.name,
                     span.status, span.duration_ms, json.dumps(span.attributes), span.error_type)
                    for span_index, span in enumerate(trace.spans)
                ]
                if span_rows:
                    await connection.execute(
                        """
                        INSERT INTO spans (
                            trace_id,
                            span_index,
                            span_id,
                            parent_span_id,
                            name,
                            status,
                            duration_ms,
                            attributes,
                            error_type
                        )
                        SELECT $1, batch.*
                        FROM unnest(
                            $2::integer[], $3::text[], $4::text[], $5::text[],
                            $6::text[], $7::double precision[], $8::jsonb[], $9::text[]
                        ) AS batch(span_index, span_id, parent_span_id, name,
                                   status, duration_ms, attributes, error_type)
                        """,
                        trace.trace_id,
                        *zip(*span_rows),
                    )

    async def get(
        self,
        trace_id: str,
    ) -> TraceRecord | None:
        async with self._pool.acquire() as connection:
            # Both SELECTs must observe one committed version, including
            # when another request replaces or deletes the trace in between.
            async with connection.transaction(isolation="repeatable_read", readonly=True):
                trace_row = await connection.fetchrow(
                    """
                    SELECT
                        trace_id,
                        request_id,
                        status,
                        duration_ms
                    FROM traces
                    WHERE trace_id = $1
                    """,
                    trace_id,
                )

                if trace_row is None:
                    return None

                span_rows = await connection.fetch(
                    """
                    SELECT
                        span_id,
                        parent_span_id,
                        name,
                        status,
                        duration_ms,
                        attributes,
                        error_type
                    FROM spans
                    WHERE trace_id = $1
                    ORDER BY span_index
                    """,
                    trace_id,
                )

        return TraceRecord(
            trace_id=trace_row["trace_id"],
            request_id=trace_row["request_id"],
            status=trace_row["status"],
            duration_ms=trace_row["duration_ms"],
            spans=[
                SpanRecord(
                    span_id=span_row["span_id"],
                    parent_span_id=span_row["parent_span_id"],
                    name=span_row["name"],
                    status=span_row["status"],
                    duration_ms=span_row["duration_ms"],
                    attributes=json.loads(span_row["attributes"]),
                    error_type=span_row["error_type"],
                )
                for span_row in span_rows
            ],
        )
