"""HTTP boundary tests; no real model, database, or PDF parser is called."""
from types import SimpleNamespace
from uuid import UUID

import asyncio
import asyncpg
import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.rag.answer import (AnswerConfigurationError, AnswerConnectionError,
                            AnswerResponseError, AnswerTimeoutError)
from app.rag.answer_models import PaperAnswer
from app.rag.embedding import (EmbeddingConnectionError, EmbeddingResponseError,
                              EmbeddingTimeoutError)
from app.rag.evidence import CitationValidationError
from app.rag.pdf import PdfReadError
from app.rag.preparation import NoUsableTextError
from app.rag.store import (EmbeddingMismatchError, PaperAlreadyExistsError,
                           PaperNotFoundError)


class FakeService:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    async def import_pdf(self, source, filename, title, **context):
        self.calls.append(("import", source.read(), filename, title, context))
        if self.error:
            raise self.error
        return SimpleNamespace(paper_id="paper-1", title="A paper", page_count=2,
                               chunk_count=4)

    async def ask(self, paper_id, question, **context):
        self.calls.append(("ask", paper_id, question, context))
        if self.error:
            raise self.error
        return PaperAnswer(paper_id=paper_id, answer="Need more evidence", claims=(),
                           citations=(), insufficient_evidence=True, reason="No context")


def make_client(service=None):
    # Deliberately avoid app.main/lifespan: these tests exercise the route boundary.
    from app.api.routes.papers import router

    app = FastAPI()
    if service is not None:
        app.state.rag_service = service
    app.include_router(router)

    @app.middleware("http")
    async def context(request: Request, call_next):
        request.state.request_id = request.headers.get("X-Request-ID", "test-request")
        response = await call_next(request)
        if hasattr(request.state, "trace_id"):
            response.headers["X-Trace-ID"] = request.state.trace_id
        return response

    return TestClient(app, raise_server_exceptions=False)


def test_import_returns_counts_and_passes_binary_file_and_context():
    service = FakeService()
    response = make_client(service).post(
        "/papers", files={"file": ("study.pdf", b"%PDF-test", "application/pdf")},
        data={"title": "Research"}, headers={"X-Request-ID": "request-123"})
    assert response.status_code == 201
    body = response.json()
    trace_id = body.pop("trace_id")
    UUID(trace_id)
    assert body == {"paper_id": "paper-1", "title": "A paper", "page_count": 2,
                    "chunk_count": 4}
    assert service.calls == [("import", b"%PDF-test", "study.pdf", "Research",
                              {"request_id": "request-123", "trace_id": trace_id})]
    assert response.headers["X-Trace-ID"] == trace_id


def test_import_allows_missing_title_and_closes_uploaded_file(monkeypatch):
    from starlette.datastructures import UploadFile
    closed = []
    original_close = UploadFile.close

    async def close(upload):
        closed.append(upload.filename)
        await original_close(upload)

    monkeypatch.setattr(UploadFile, "close", close)
    service = FakeService()
    response = make_client(service).post("/papers", files={"file": ("a.pdf", b"pdf")})
    assert response.status_code == 201
    assert service.calls[0][3] is None
    assert "a.pdf" in closed


def test_question_returns_flat_answer_and_uses_fresh_trace_each_time():
    service = FakeService()
    client = make_client(service)
    responses = [client.post("/papers/paper-1/questions", json={"question": "  Why?  "})
                 for _ in range(2)]
    assert all(response.status_code == 200 for response in responses)
    traces = [response.json()["trace_id"] for response in responses]
    assert traces[0] != traces[1]
    for trace_id, response, call in zip(traces, responses, service.calls):
        UUID(trace_id)
        assert response.headers["X-Trace-ID"] == trace_id
        assert response.json()["paper_id"] == "paper-1"
        assert response.json()["answer"] == "Need more evidence"
        assert response.json()["claims"] == []
        assert response.json()["citations"] == []
        assert "result" not in response.json()
        assert call == ("ask", "paper-1", "Why?", {"top_k": 5,
                        "request_id": "test-request", "trace_id": trace_id})


@pytest.mark.parametrize("payload", [
    {"question": ""}, {"question": " \n "}, {"question": "x" * 4001},
    {"question": "Q", "top_k": 0}, {"question": "Q", "top_k": 101},
    {"question": "Q", "top_k": 1.1}, {"question": "Q", "top_k": True},
    {"question": "Q", "top_k": "2"},
])
def test_invalid_question_is_rejected_before_service(payload):
    service = FakeService()
    response = make_client(service).post("/papers/paper-1/questions", json=payload)
    assert response.status_code == 422
    assert service.calls == []


@pytest.mark.parametrize("top_k", [1, 100])
def test_question_accepts_top_k_bounds(top_k):
    service = FakeService()
    response = make_client(service).post("/papers/paper-1/questions",
                                        json={"question": "Q", "top_k": top_k})
    assert response.status_code == 200
    assert service.calls[0][3]["top_k"] == top_k


@pytest.mark.parametrize("method,path,kwargs", [
    ("post", "/papers", {"files": {"file": ("a.pdf", b"pdf")}}),
    ("post", "/papers/a/questions", {"json": {"question": "Q"}}),
])
def test_unconfigured_service_is_503(method, path, kwargs):
    response = getattr(make_client(), method)(path, **kwargs)
    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"].lower()


def error_cases():
    from app.rag.service_errors import (RagBusyError, RagInputError, RagLimitError,
                                        RagUnavailableError)
    return [
        (RagInputError, 400), (PdfReadError, 400), (NoUsableTextError, 400),
        (RagLimitError, 413), (PaperNotFoundError, 404),
        (PaperAlreadyExistsError, 409), (EmbeddingMismatchError, 409),
        (EmbeddingTimeoutError, 504), (AnswerTimeoutError, 504),
        (EmbeddingConnectionError, 503), (AnswerConnectionError, 503),
        (RagBusyError, 503), (RagUnavailableError, 503),
        (AnswerConfigurationError, 503), (asyncpg.PostgresError, 503), (OSError, 503),
        (EmbeddingResponseError, 502), (AnswerResponseError, 502),
        (CitationValidationError, 502),
    ]


@pytest.mark.parametrize("endpoint", ["import", "ask"])
def test_known_errors_have_safe_status_and_trace(endpoint):
    secret = "private remote response /home/user/secret.pdf token=secret"
    for error_type, expected_status in error_cases():
        service = FakeService(error_type(secret))
        client = make_client(service)
        response = (client.post("/papers", files={"file": ("a.pdf", b"pdf")})
                    if endpoint == "import" else
                    client.post("/papers/a/questions", json={"question": "Q"}))
        assert response.status_code == expected_status, (error_type, response.text)
        assert secret not in response.text
        assert "secret" not in response.text
        UUID(response.headers["X-Trace-ID"])


def test_openapi_exposes_papers_tag():
    schema = make_client(FakeService()).get("/openapi.json").json()
    assert "papers" in schema["paths"]["/papers"]["post"]["tags"]
    assert "papers" in schema["paths"]["/papers/{paper_id}/questions"]["post"]["tags"]


def test_streamed_multipart_without_content_length_is_rejected_before_service():
    from app.api.rag_upload_limit import RagUploadLimitMiddleware

    service = FakeService()
    app = make_client(service).app
    app.add_middleware(RagUploadLimitMiddleware,
                       settings=SimpleNamespace(rag_max_file_bytes=8))
    produced = []

    async def body():
        chunks = [
            b"--boundary\r\nContent-Disposition: form-data; name=\"file\"; "
            b"filename=\"a.pdf\"\r\nContent-Type: application/pdf\r\n\r\n" + b"a" * 40000,
            b"b" * 30000,
            b"\r\n--boundary--\r\n",
        ]
        for index, chunk in enumerate(chunks):
            produced.append(index)
            yield chunk

    async def send():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                    base_url="http://test") as client:
            return await client.post("/papers", content=body(),
                                     headers={"Content-Type": "multipart/form-data; boundary=boundary"})

    response = asyncio.run(send())
    assert response.request.headers.get("content-length") is None
    assert response.status_code == 413
    assert service.calls == []
    assert produced == [0, 1]
