from importlib.util import find_spec

import pytest


def test_health_returns_ok():
    """GET /health should return HTTP 200 and a small JSON status body."""
    if find_spec("app.main") is None:
        pytest.fail("app/main.py 尚未实现；现在进入 TDD 红灯阶段")

    from fastapi.testclient import TestClient
    from app.main import app

    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
