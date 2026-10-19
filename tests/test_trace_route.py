import asyncio
from uuid import uuid4

from fastapi.testclient import TestClient

from app.api.dependencies import get_trace_store
from app.main import app
from app.observability.models import TraceRecord


def test_get_trace_returns_saved_trace():
    trace = TraceRecord(
        trace_id=str(uuid4()),
        request_id="request-query-test",
        status="ok",
        duration_ms=1.0,
        spans=[],
    )

    store = get_trace_store()

    asyncio.run(store.save(trace))

    with TestClient(app) as client:
        response = client.get(
            f"/traces/{trace.trace_id}"
        )

    assert response.status_code == 200
    assert response.json() == trace.model_dump(
        mode="json"
    )


def test_get_trace_returns_404_for_unknown_trace_id():
    missing_id = str(uuid4())

    with TestClient(app) as client:
        response = client.get(
            f"/traces/{missing_id}"
        )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Trace not found",
    }
