import asyncio
import json

import httpx
import pytest


def make_embedding(client, **overrides):
    from app.rag.embedding import OllamaEmbeddingClient

    options = dict(client=client, model="qwen3-embedding:0.6b",
                   embed_url="http://ollama:11434/api/embed", dimensions=3,
                   batch_size=2, timeout_seconds=7.5,
                   query_instruction="Retrieve scientific evidence.")
    options.update(overrides)
    return OllamaEmbeddingClient(**options)


def response(vectors, *, model="qwen3-embedding:0.6b"):
    return httpx.Response(200, content=json.dumps({"model": model, "embeddings": vectors}))


def test_documents_batch_in_order_with_no_truncation_and_request_timeout():
    requests = []

    def handler(request):
        requests.append(request)
        texts = json.loads(request.content)["input"]
        return response([[int(text), 1, 2] for text in texts])

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=99) as client:
            docs = ["1", "2", "3", "4", "5"]
            result = await make_embedding(client).embed_documents(docs)
            assert not client.is_closed
            assert docs == ["1", "2", "3", "4", "5"]
            return result

    assert asyncio.run(run()) == [[float(i), 1.0, 2.0] for i in range(1, 6)]
    assert [json.loads(r.content)["input"] for r in requests] == [["1", "2"], ["3", "4"], ["5"]]
    for request in requests:
        assert request.method == "POST"
        assert str(request.url) == "http://ollama:11434/api/embed"
        body = json.loads(request.content)
        assert body == {"model": "qwen3-embedding:0.6b", "input": body["input"],
                        "truncate": False, "dimensions": 3}
        assert request.extensions["timeout"] == dict(connect=7.5, read=7.5, write=7.5, pool=7.5)


def test_query_has_instruction_but_document_text_is_unchanged():
    inputs = []

    def handler(request):
        inputs.append(json.loads(request.content)["input"])
        return response([[0.1, 0.2, 0.3]])

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            embedding = make_embedding(client)
            assert await embedding.embed_query("缓存如何工作？") == [0.1, 0.2, 0.3]
            await embedding.embed_documents(["  Original paper passage.  "])

    asyncio.run(run())
    assert inputs == [["Instruct: Retrieve scientific evidence.\nQuery:缓存如何工作？"],
                      ["  Original paper passage.  "]]


def test_empty_documents_do_not_call_service():
    async def run():
        def fail(request):
            pytest.fail("Empty input must not send an HTTP request")
        async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as client:
            assert await make_embedding(client).embed_documents([]) == []
    asyncio.run(run())


@pytest.mark.parametrize("texts", [[""], [" \t\n"], [None], [123], "not-a-list", ["ok", "ok", " "]])
def test_all_document_inputs_validate_before_any_request(texts):
    async def run():
        def fail(request):
            pytest.fail("Invalid inputs must fail before even the first batch")
        async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as client:
            with pytest.raises(ValueError):
                await make_embedding(client).embed_documents(texts)
    asyncio.run(run())


@pytest.mark.parametrize("query", ["", " \n", None, 1])
def test_invalid_query_is_rejected(query):
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: pytest.fail("No HTTP"))) as client:
            with pytest.raises(ValueError):
                await make_embedding(client).embed_query(query)
    asyncio.run(run())


@pytest.mark.parametrize("payload", [
    None, [], {}, {"model": "qwen3-embedding:0.6b"},
    {"model": "another-model", "embeddings": [[1, 2, 3]]},
    {"model": "qwen3-embedding:0.6b", "embeddings": []},
    {"model": "qwen3-embedding:0.6b", "embeddings": [[1, 2, 3], [4, 5, 6]]},
    {"model": "qwen3-embedding:0.6b", "embeddings": "invalid"},
    {"model": "qwen3-embedding:0.6b", "embeddings": [[1, 2]]},
    {"model": "qwen3-embedding:0.6b", "embeddings": [[]]},
    {"model": "qwen3-embedding:0.6b", "embeddings": [[0, 0, 0]]},
    {"model": "qwen3-embedding:0.6b", "embeddings": [[True, 2, 3]]},
    {"model": "qwen3-embedding:0.6b", "embeddings": [["1", 2, 3]]},
    {"model": "qwen3-embedding:0.6b", "embeddings": [[None, 2, 3]]},
    {"model": "qwen3-embedding:0.6b", "embeddings": [[float("nan"), 2, 3]]},
    {"model": "qwen3-embedding:0.6b", "embeddings": [[float("inf"), 2, 3]]},
])
def test_invalid_response_is_never_returned_as_vectors(payload):
    from app.rag.embedding import EmbeddingResponseError

    async def run():
        handler = lambda r: httpx.Response(200, content=json.dumps(payload))
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(EmbeddingResponseError):
                await make_embedding(client).embed_documents(["passage"])
            assert not client.is_closed
    asyncio.run(run())


def test_invalid_json_preserves_cause():
    from app.rag.embedding import EmbeddingResponseError

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, text="not json"))) as client:
            with pytest.raises(EmbeddingResponseError) as caught:
                await make_embedding(client).embed_query("query")
            assert isinstance(caught.value.__cause__, ValueError)
    asyncio.run(run())


@pytest.mark.parametrize("status", [400, 404, 429, 500])
def test_http_error_has_status_and_is_not_retried(status):
    from app.rag.embedding import EmbeddingResponseError
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json={"error": "remote error"})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(EmbeddingResponseError) as caught:
                await make_embedding(client).embed_query("query")
            assert caught.value.status_code == status
            assert isinstance(caught.value.__cause__, httpx.HTTPStatusError)
            assert not client.is_closed
    asyncio.run(run())
    assert len(calls) == 1


@pytest.mark.parametrize("exception,expected", [
    (httpx.ReadTimeout, "EmbeddingTimeoutError"),
    (httpx.ConnectError, "EmbeddingConnectionError"),
    (httpx.RemoteProtocolError, "EmbeddingConnectionError"),
])
def test_transport_errors_are_translated_with_cause(exception, expected):
    from app.rag import embedding as module

    def handler(request):
        raise exception("simulated", request=request)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(getattr(module, expected)) as caught:
                await make_embedding(client).embed_query("query")
            assert isinstance(caught.value.__cause__, exception)
    asyncio.run(run())


def test_failed_later_batch_returns_no_partial_results_or_extra_requests():
    from app.rag.embedding import EmbeddingResponseError
    calls = []

    def handler(request):
        calls.append(request)
        return response([[1, 2, 3], [4, 5, 6]]) if len(calls) == 1 else httpx.Response(500)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(EmbeddingResponseError):
                await make_embedding(client).embed_documents(["a", "b", "c", "d", "e"])
    asyncio.run(run())
    assert len(calls) == 2


def test_request_cancellation_propagates():
    def handler(request):
        raise asyncio.CancelledError()

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(asyncio.CancelledError):
                await make_embedding(client).embed_query("query")
    asyncio.run(run())


@pytest.mark.parametrize("options", [
    {"dimensions": 0}, {"dimensions": True}, {"dimensions": 1.5},
    {"batch_size": 0}, {"batch_size": True}, {"batch_size": 1.5},
    {"timeout_seconds": 0}, {"timeout_seconds": -1}, {"timeout_seconds": float("inf")},
    {"timeout_seconds": float("nan")}, {"timeout_seconds": True},
    {"model": " "}, {"query_instruction": "\n"},
])
def test_client_configuration_is_validated(options):
    async def run():
        async with httpx.AsyncClient() as client:
            with pytest.raises(ValueError):
                make_embedding(client, **options)
    asyncio.run(run())
