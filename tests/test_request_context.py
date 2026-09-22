from fastapi.testclient import TestClient

from app.main import app


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