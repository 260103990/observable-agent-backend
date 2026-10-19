from fastapi import Depends

from app.agent.ollama_decision_maker import (
    OllamaDecisionMaker,
)
from app.agent.runner import AgentRunner
from app.config import Settings, get_settings
from app.tools.calculator import ADD_NUMBERS_TOOL
from app.tools.registry import ToolRegistry
from app.observability.store import (
    InMemoryTraceStore,
    TraceStore,
)


_trace_store = InMemoryTraceStore()


def get_trace_store() -> TraceStore:
    return _trace_store


def get_agent_runner(
    settings: Settings = Depends(get_settings),
    trace_store: TraceStore = Depends(get_trace_store),
) -> AgentRunner:
    registry = ToolRegistry()
    registry.register(ADD_NUMBERS_TOOL)

    decision_maker = OllamaDecisionMaker(
        model=settings.ollama_model,
        chat_url=settings.ollama_chat_url,
        timeout_seconds=settings.ollama_timeout_seconds,
        client=None,
    )

    return AgentRunner(
        registry=registry,
        decision_maker=decision_maker,
        trace_store=trace_store,
    )
