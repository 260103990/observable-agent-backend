import asyncio
import json
from types import SimpleNamespace

import pytest


def exercise(*, chunks, content_length=None, path="/papers", method="POST", limit=8):
    from app.api.rag_upload_limit import RagUploadLimitMiddleware

    received = []
    sent = []
    settings = SimpleNamespace(rag_max_file_bytes=limit)

    async def inner(scope, receive, send):
        body = b""
        while True:
            message = await receive()
            received.append(message)
            body += message.get("body", b"")
            if not message.get("more_body", False):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": str(len(body)).encode()})

    messages = [{"type": "http.request", "body": chunk,
                 "more_body": i < len(chunks) - 1} for i, chunk in enumerate(chunks)]
    read_count = 0

    async def receive():
        nonlocal read_count
        read_count += 1
        return messages.pop(0)

    async def send(message):
        sent.append(message)

    headers = ([] if content_length is None else
               [(b"content-length", str(content_length).encode())])
    scope = {"type": "http", "method": method, "path": path, "headers": headers}
    middleware = RagUploadLimitMiddleware(inner, settings=settings)
    asyncio.run(middleware(scope, receive, send))
    return sent, received, read_count


def test_content_length_rejects_oversize_before_reading_body():
    sent, received, reads = exercise(chunks=[b"x"], content_length=65545)
    assert sent[0]["status"] == 413
    assert reads == 0
    assert received == []
    assert "detail" in json.loads(sent[1]["body"])


def test_stream_without_content_length_is_cumulatively_limited():
    sent, received, reads = exercise(chunks=[b"a" * 40000, b"b" * 30000, b"later"])
    assert sent[0]["status"] == 413
    assert reads == 2
    assert len(received) == 1


def test_false_small_content_length_cannot_bypass_stream_limit():
    sent, _, reads = exercise(chunks=[b"a" * 65544, b"b"], content_length=1)
    assert sent[0]["status"] == 413
    assert reads == 2


def test_body_at_total_limit_is_forwarded_unmodified():
    chunks = [b"a" * 65536, b"b" * 8]
    sent, received, reads = exercise(chunks=chunks)
    assert sent[0]["status"] == 200
    assert sent[1]["body"] == b"65544"
    assert [message["body"] for message in received] == chunks
    assert reads == 2


@pytest.mark.parametrize("path,method", [("/agent/run", "POST"),
                                         ("/papers/a/questions", "POST"),
                                         ("/papers", "GET")])
def test_other_routes_and_methods_are_untouched(path, method):
    sent, received, _ = exercise(chunks=[b"a" * 70000], content_length=70000,
                                path=path, method=method)
    assert sent[0]["status"] == 200
    assert len(received[0]["body"]) == 70000
