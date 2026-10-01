import logging
from io import StringIO

from fastapi.testclient import TestClient

from app.main import app, request_logger

client = TestClient(app)


def test_response_includes_generated_request_id():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.headers.get("x-request-id")


def test_response_preserves_client_request_id():
    response = client.get(
        "/health",
        headers={
            "X-Request-ID": "client-request-123",
        },
    )

    assert response.status_code == 200
    assert (
        response.headers["x-request-id"]
        == "client-request-123"
    )


def test_response_includes_process_time():
    response = client.get("/health")

    assert response.status_code == 200

    process_time = response.headers.get(
        "x-process-time-ms"
    )

    assert process_time is not None
    assert float(process_time) >= 0


def test_completed_request_is_logged(caplog):
    request_logger.addHandler(caplog.handler)

    try:
        with caplog.at_level(
            logging.INFO,
            logger="app.request",
        ):
            response = client.get(
                "/health",
                headers={
                    "X-Request-ID": "log-test-123",
                },
            )
    finally:
        request_logger.removeHandler(caplog.handler)
    assert response.status_code == 200

    request_logs = [
        record
        for record in caplog.records
        if record.name == "app.request"
    ]

    assert len(request_logs) == 1

    record = request_logs[0]

    assert record.getMessage() == "request_completed"
    assert record.request_id == "log-test-123"
    assert record.method == "GET"
    assert record.path == "/health"
    assert record.status_code == 200
    assert record.process_time_ms >= 0


def test_unhandled_exception_is_logged(
    monkeypatch,
    caplog,
):
    async def fake_generate(
        model: str,
        prompt: str,
        generate_url: str,
        timeout_seconds: float,
    ) -> str:
        raise RuntimeError("unexpected failure")

    monkeypatch.setattr(
        "app.api.routes.chat.generate",
        fake_generate,
    )

    error_client = TestClient(
        app,
        raise_server_exceptions=False,
    )

    request_logger.addHandler(caplog.handler)

    try:
        with caplog.at_level(
            logging.ERROR,
            logger="app.request",
        ):
            response = error_client.post(
                "/chat",
                headers={
                    "X-Request-ID": "failed-request-123",
                },
                json={
                    "message": "Hello",
                },
            )
    finally:
        request_logger.removeHandler(caplog.handler)

    assert response.status_code == 500

    failure_logs = [
        record
        for record in caplog.records
        if (
            record.name == "app.request"
            and record.getMessage() == "request_failed"
        )
    ]

    assert len(failure_logs) == 1

    record = failure_logs[0]

    assert record.levelno == logging.ERROR
    assert record.request_id == "failed-request-123"
    assert record.method == "POST"
    assert record.path == "/chat"
    assert record.status_code == 500
    assert record.exception_type == "RuntimeError"
    assert record.process_time_ms >= 0
    formatted_error = (
        request_logger.handlers[0]
        .formatter
        .format(record)
    )

    assert (
        "exception_type=RuntimeError"
        in formatted_error
    )


def test_request_logger_writes_structured_output(
    monkeypatch,
):
    output_stream = StringIO()

    stream_handlers = [
        handler
        for handler in request_logger.handlers
        if isinstance(handler, logging.StreamHandler)
    ]

    assert len(stream_handlers) == 1

    monkeypatch.setattr(
        stream_handlers[0],
        "stream",
        output_stream,
    )

    request_logger.info(
        "request_completed",
        extra={
            "request_id": "runtime-test-123",
            "method": "GET",
            "path": "/health",
            "status_code": 200,
            "process_time_ms": 1.25,
        },
    )

    output = output_stream.getvalue()

    assert "request_completed" in output
    assert "request_id=runtime-test-123" in output
    assert "method=GET" in output
    assert "path=/health" in output
    assert "status_code=200" in output
    assert "process_time_ms=1.25" in output


def test_request_logger_does_not_duplicate_to_root(
    monkeypatch,
):
    request_output = StringIO()
    root_output = StringIO()

    stream_handlers = [
        handler
        for handler in request_logger.handlers
        if isinstance(handler, logging.StreamHandler)
    ]

    assert len(stream_handlers) == 1

    monkeypatch.setattr(
        stream_handlers[0],
        "stream",
        request_output,
    )

    root_handler = logging.StreamHandler(root_output)
    root_handler.setLevel(logging.INFO)

    root_logger = logging.getLogger()
    root_logger.addHandler(root_handler)

    try:
        request_logger.info(
            "request_completed",
            extra={
                "request_id": "duplicate-test-123",
                "method": "GET",
                "path": "/health",
                "status_code": 200,
                "process_time_ms": 1.25,
            },
        )
    finally:
        root_logger.removeHandler(root_handler)

    assert "request_completed" in request_output.getvalue()
    assert root_output.getvalue() == ""