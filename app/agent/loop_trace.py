"""Loop spans contain counters and safe codes, never prompts or tool payloads."""
import asyncio
from contextlib import contextmanager
import logging
from time import perf_counter
from uuid import uuid4

from app.observability.models import SpanRecord, TraceRecord


logger = logging.getLogger('app.agent.loop')


class LoopTrace:
    def __init__(self, store, request_id, trace_id, *, timeout_seconds=5.0, root_name="agent.loop"):
        if not request_id or not trace_id:
            raise ValueError('Loop traces require request_id and trace_id')
        self.store, self.request_id, self.trace_id = store, request_id, trace_id
        self.root_id, self.attributes, self.spans = str(uuid4()), {}, []
        self.timeout_seconds = timeout_seconds
        self.root_name = root_name

    async def __aenter__(self):
        self.started = perf_counter()
        return self

    @contextmanager
    def stage(self, name, **attributes):
        started = perf_counter()
        status, error = 'ok', None
        try:
            yield attributes
        except BaseException as exc:
            status = 'cancelled' if isinstance(exc, asyncio.CancelledError) else 'error'
            error = type(exc).__name__
            raise
        finally:
            # A recoverable tool failure is an error span, while the loop may finish successfully.
            error = error or attributes.pop('error_code', None)
            if error and status == 'ok':
                status = 'error'
            self.spans.append(SpanRecord(span_id=str(uuid4()), parent_span_id=self.root_id,
                name=name, status=status, error_type=error,
                duration_ms=(perf_counter()-started)*1000, attributes=attributes))

    async def __aexit__(self, exc_type, exc, tb):
        status = 'ok' if exc is None else ('cancelled' if isinstance(exc, asyncio.CancelledError) else 'error')
        duration = (perf_counter()-self.started)*1000
        root = SpanRecord(span_id=self.root_id, parent_span_id=None, name=self.root_name, status=status,
            error_type=None if exc is None else type(exc).__name__, duration_ms=duration, attributes=self.attributes)
        try:
            async with asyncio.timeout(self.timeout_seconds):
                await self.store.save(TraceRecord(trace_id=self.trace_id, request_id=self.request_id,
                    status=status, duration_ms=duration, spans=[root, *self.spans]))
        except Exception as error:
            logger.warning('loop.trace_save_failed error_type=%s', type(error).__name__)
        return False
