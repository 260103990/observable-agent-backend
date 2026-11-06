"""Small RAG-specific trace scope; the existing Agent trace contract is intact."""
import asyncio
from contextlib import contextmanager
import logging
from time import perf_counter
from uuid import uuid4

from app.observability.models import SpanRecord, TraceRecord


logger = logging.getLogger("app.rag")


class RagRun:
    def __init__(self, store, name: str, request_id: str, trace_id: str):
        if not request_id or not trace_id:
            raise ValueError("RAG traces require request_id and trace_id")
        self.store, self.name = store, name
        self.request_id, self.trace_id = request_id, trace_id
        self.root_id = str(uuid4())
        self.attributes = {}
        self.spans = []

    async def __aenter__(self):
        self.started = perf_counter()
        return self

    @contextmanager
    def stage(self, name: str, **attributes):
        started = perf_counter()
        status, error_type = "ok", None
        try:
            yield attributes
        except BaseException as exc:
            status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "error"
            error_type = type(exc).__name__
            raise
        finally:
            self.spans.append(SpanRecord(span_id=str(uuid4()), parent_span_id=self.root_id, name=name,
                status=status, error_type=error_type, duration_ms=(perf_counter()-started)*1000,
                attributes=attributes))

    async def __aexit__(self, exc_type, exc, tb):
        status = "ok" if exc is None else ("cancelled" if isinstance(exc, asyncio.CancelledError) else "error")
        duration = (perf_counter()-self.started)*1000
        root = SpanRecord(span_id=self.root_id,parent_span_id=None,name=self.name,status=status,
            error_type=None if exc is None else type(exc).__name__,duration_ms=duration,attributes=self.attributes)
        trace = TraceRecord(trace_id=self.trace_id,request_id=self.request_id,status=status,
                            duration_ms=duration,spans=[root,*self.spans])
        try:
            await self.store.save(trace)
        except Exception as error:
            # Log the type only; preserving the business result/original error is deliberate.
            logger.warning("rag.trace_save_failed error_type=%s",type(error).__name__)
        return False
