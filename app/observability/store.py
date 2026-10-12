from app.observability.models import TraceRecord
from typing import Protocol
from asyncio import Lock


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
