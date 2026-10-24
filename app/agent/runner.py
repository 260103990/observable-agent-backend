from collections.abc import Awaitable, Callable
from time import perf_counter
from uuid import uuid4
import logging
from app.observability.models import SpanRecord, TraceRecord
from app.observability.store import TraceStore
from app.schemas.agent import AgentRunResult, ToolDecision
from app.tools.registry import ToolRegistry


agent_logger = logging.getLogger("app.agent")


DecisionMaker = Callable[
    [str, list[dict[str, object]]],
    Awaitable[ToolDecision],
]

Summarizer = Callable[
    [str, ToolDecision, object],
    Awaitable[str],
]

class AgentRunner:
    def __init__(
        self,
        registry: ToolRegistry,
        decision_maker: DecisionMaker,
        summarizer: Summarizer,
        trace_store: TraceStore | None = None,
    ) -> None:
        self._registry = registry
        self._decision_maker = decision_maker
        self._summarizer = summarizer
        self._trace_store = trace_store

    async def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        trace_id: str | None = None,
    ) -> AgentRunResult:
        if self._trace_store is not None:
            if request_id is None or trace_id is None:
                raise ValueError(
                    "Traced runs require request_id and trace_id"
                )

        started_at = perf_counter()

        tool_schemas = self._registry.list_schemas()

        model_started_at = perf_counter()

        try:
            decision = await self._decision_maker(
                prompt,
                tool_schemas,
            )
            tool_call = decision.tool_call
        except Exception as exc:
            if self._trace_store is not None:
                finished_at = perf_counter()

                model_duration_ms = (
                    finished_at - model_started_at
                ) * 1000

                duration_ms = (
                    finished_at - started_at
                ) * 1000

                error_type = type(exc).__name__

                root_span = SpanRecord(
                    span_id=str(uuid4()),
                    parent_span_id=None,
                    name="agent.run",
                    status="error",
                    duration_ms=duration_ms,
                    attributes={},
                    error_type=error_type,
                )

                model_span = SpanRecord(
                    span_id=str(uuid4()),
                    parent_span_id=root_span.span_id,
                    name="model.decision",
                    status="error",
                    duration_ms=model_duration_ms,
                    attributes={
                        "available_tool_count": len(tool_schemas),
                    },
                    error_type=error_type,
                )

                trace = TraceRecord(
                    trace_id=trace_id,
                    request_id=request_id,
                    status="error",
                    duration_ms=duration_ms,
                    spans=[root_span, model_span],
                )

                await self._save_trace(trace)

            raise

        model_duration_ms = (
            perf_counter() - model_started_at
        ) * 1000

        tool_started_at = perf_counter()

        try:
            result = self._registry.execute(
                tool_call.name,
                tool_call.arguments,
            )
        except Exception as exc:
            if self._trace_store is not None:
                finished_at = perf_counter()

                tool_duration_ms = (
                    finished_at - tool_started_at
                ) * 1000

                duration_ms = (
                    finished_at - started_at
                ) * 1000

                error_type = type(exc).__name__

                root_span = SpanRecord(
                    span_id=str(uuid4()),
                    parent_span_id=None,
                    name="agent.run",
                    status="error",
                    duration_ms=duration_ms,
                    attributes={},
                    error_type=error_type,
                )

                model_span = SpanRecord(
                    span_id=str(uuid4()),
                    parent_span_id=root_span.span_id,
                    name="model.decision",
                    status="ok",
                    duration_ms=model_duration_ms,
                    attributes={
                        "available_tool_count": len(tool_schemas),
                    },
                    error_type=None,
                )

                tool_span = SpanRecord(
                    span_id=str(uuid4()),
                    parent_span_id=root_span.span_id,
                    name="tool.execute",
                    status="error",
                    duration_ms=tool_duration_ms,
                    attributes={
                        "tool_name": tool_call.name,
                    },
                    error_type=error_type,
                )

                trace = TraceRecord(
                    trace_id=trace_id,
                    request_id=request_id,
                    status="error",
                    duration_ms=duration_ms,
                    spans=[
                        root_span,
                        model_span,
                        tool_span,
                    ],
                )

                await self._save_trace(trace)

            raise

        tool_duration_ms = (
            perf_counter() - tool_started_at
        ) * 1000

        summary_started_at = perf_counter()

        try:
            answer = await self._summarizer(
                prompt,
                decision,
                result,
            )
        except Exception as exc:
            if self._trace_store is not None:
                finished_at = perf_counter()

                summary_duration_ms = (
                    finished_at - summary_started_at
                ) * 1000

                duration_ms = (
                    finished_at - started_at
                ) * 1000

                error_type = type(exc).__name__

                root_span = SpanRecord(
                    span_id=str(uuid4()),
                    parent_span_id=None,
                    name="agent.run",
                    status="error",
                    duration_ms=duration_ms,
                    attributes={},
                    error_type=error_type,
                )

                model_span = SpanRecord(
                    span_id=str(uuid4()),
                    parent_span_id=root_span.span_id,
                    name="model.decision",
                    status="ok",
                    duration_ms=model_duration_ms,
                    attributes={
                        "available_tool_count": len(tool_schemas),
                    },
                    error_type=None,
                )

                tool_span = SpanRecord(
                    span_id=str(uuid4()),
                    parent_span_id=root_span.span_id,
                    name="tool.execute",
                    status="ok",
                    duration_ms=tool_duration_ms,
                    attributes={
                        "tool_name": tool_call.name,
                    },
                    error_type=None,
                )

                summary_span = SpanRecord(
                    span_id=str(uuid4()),
                    parent_span_id=root_span.span_id,
                    name="model.summary",
                    status="error",
                    duration_ms=summary_duration_ms,
                    attributes={},
                    error_type=error_type,
                )

                trace = TraceRecord(
                    trace_id=trace_id,
                    request_id=request_id,
                    status="error",
                    duration_ms=duration_ms,
                    spans=[
                        root_span,
                        model_span,
                        tool_span,
                        summary_span,
                    ],
                )

                await self._save_trace(trace)

            raise
        summary_duration_ms = (
            perf_counter() - summary_started_at
        ) * 1000
        if self._trace_store is not None:
            duration_ms = (
                perf_counter() - started_at
            ) * 1000

            root_span = SpanRecord(
                span_id=str(uuid4()),
                parent_span_id=None,
                name="agent.run",
                status="ok",
                duration_ms=duration_ms,
                attributes={},
                error_type=None,
            )

            model_span = SpanRecord(
                span_id=str(uuid4()),
                parent_span_id=root_span.span_id,
                name="model.decision",
                status="ok",
                duration_ms=model_duration_ms,
                attributes={
                    "available_tool_count": len(tool_schemas),
                },
                error_type=None,
            )

            tool_span = SpanRecord(
                span_id=str(uuid4()),
                parent_span_id=root_span.span_id,
                name="tool.execute",
                status="ok",
                duration_ms=tool_duration_ms,
                attributes={
                    "tool_name": tool_call.name,
                },
                error_type=None,
            )

            summary_span = SpanRecord(
                span_id=str(uuid4()),
                parent_span_id=root_span.span_id,
                name="model.summary",
                status="ok",
                duration_ms=summary_duration_ms,
                attributes={},
                error_type=None,
            )

            trace = TraceRecord(
                trace_id=trace_id,
                request_id=request_id,
                status="ok",
                duration_ms=duration_ms,
                spans=[
                    root_span,
                    model_span,
                    tool_span,
                    summary_span,
                ],
            )

            await self._save_trace(trace)

        return AgentRunResult(
            result=result,
            answer=answer,
        )

    async def _save_trace(
        self,
        trace: TraceRecord,
    ) -> None:
        if self._trace_store is None:
            return

        try:
            await self._trace_store.save(trace)
        except Exception as exc:
            agent_logger.warning(
                "trace_save_failed",
                extra={
                    "request_id": trace.request_id,
                    "trace_id": trace.trace_id,
                    "exception_type": type(exc).__name__,
                },
            )
            return