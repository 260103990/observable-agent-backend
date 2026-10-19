from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_trace_store
from app.observability.models import TraceRecord
from app.observability.store import TraceStore


router = APIRouter(tags=["traces"])


@router.get("/traces/{trace_id}")
async def get_trace(
    trace_id: str,
    store: TraceStore = Depends(get_trace_store),
) -> TraceRecord:
    trace = await store.get(trace_id)

    if trace is None:
        raise HTTPException(
            status_code=404,
            detail="Trace not found",
        )

    return trace
