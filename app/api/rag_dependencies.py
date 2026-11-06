"""Request-scoped RAG context and application-owned service lookup."""
from uuid import uuid4

from fastapi import HTTPException, Request


def get_rag_trace_id(request: Request) -> str:
    trace_id = str(uuid4())
    request.state.trace_id = trace_id
    return trace_id


def get_rag_service(request: Request):
    service = getattr(request.app.state, "rag_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="RAG service is unavailable")
    return service
