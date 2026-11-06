import asyncio
import json

import httpx
import pytest
from pydantic import SecretStr

from tests.answer_helpers import hit


DRAFT = {"claims": [{"text": "吞吐量为42 requests/s。", "evidence_ids": ["E1"]}],
         "insufficient_evidence": False, "reason": None}


def envelope(draft=None, **overrides):
    result = {"id": "completion-1", "model": "deepseek-flash", "choices": [{
        "finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(draft or DRAFT)}}]}
    result.update(overrides)
    return result


def make_answer(client, **overrides):
    from app.rag.answer import DeepSeekAnswerClient
    options = dict(client=client, api_key=SecretStr("unit-test-key"), model="deepseek-flash",
                   chat_url="https://api.deepseek.com/chat/completions", timeout_seconds=7.5,
                   max_tokens=2048, max_evidence_chars=12000, max_evidence_items=10)
    options.update(overrides)
    return DeepSeekAnswerClient(**options)


def test_request_json_contract_auth_timeout_and_backend_citations():
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=envelope())
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=99) as http:
            result = await make_answer(http).answer("paper-1", "实验吞吐量是多少？", [hit()])
            assert not http.is_closed
            assert result.citations[0].raw_text == hit().chunk.raw_text
            assert result.generation_model == "deepseek-flash"
            assert result.provider_request_id == "completion-1"
            return result
    result = asyncio.run(run())
    assert result.answer.endswith("[E1]")
    request = requests[0]
    assert str(request.url) == "https://api.deepseek.com/chat/completions"
    assert request.headers["authorization"] == "Bearer unit-test-key"
    assert request.extensions["timeout"]["read"] == 7.5
    payload = json.loads(request.content)
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["stream"] is False and payload["max_tokens"] == 2048
    assert "json" in payload["messages"][0]["content"].lower()
    evidence = json.loads(payload["messages"][1]["content"])
    assert evidence["question"] == "实验吞吐量是多少？"
    assert evidence["evidence"][0]["evidence_id"] == "E1"
    assert evidence["evidence"][0]["text"] == hit().chunk.text


def test_empty_evidence_returns_local_insufficiency_without_key_or_request():
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: pytest.fail("No HTTP"))) as http:
            result = await make_answer(http, api_key=None).answer("paper-1", "question", [])
            assert result.insufficient_evidence and not result.citations and not result.claims
            assert result.generation_model is None
    asyncio.run(run())


def test_missing_key_for_nonempty_evidence_fails_before_http():
    from app.rag.answer import AnswerConfigurationError
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: pytest.fail("No HTTP"))) as http:
            with pytest.raises(AnswerConfigurationError):
                await make_answer(http, api_key=None).answer("paper-1", "question", [hit()])
    asyncio.run(run())


@pytest.mark.parametrize("question", ["", " \n", None])
def test_blank_question_is_rejected(question):
    async def run():
        async with httpx.AsyncClient() as http:
            with pytest.raises(ValueError):
                await make_answer(http).answer("paper-1", question, [])
    asyncio.run(run())


@pytest.mark.parametrize("payload", [None, [], {}, {"choices": []}, envelope(choices=[{}, {}]),
    envelope(choices=[{"finish_reason": "length", "message": {"role": "assistant", "content": json.dumps(DRAFT)}}]),
    envelope(choices=[{"finish_reason": "content_filter", "message": {"role": "assistant", "content": "{}"}}]),
    envelope(choices=[{"finish_reason": "stop", "message": {"role": "assistant", "content": " "}}]),
    envelope(choices=[{"finish_reason": "stop", "message": {"role": "assistant", "content": "not-json"}}]),
    envelope(choices=[{"finish_reason": "stop", "message": {"role": "tool", "content": json.dumps(DRAFT)}}]),
    envelope(choices=[{"finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(DRAFT), "tool_calls": [{}]}}]),
])
def test_invalid_or_incomplete_response_is_rejected(payload):
    from app.rag.answer import AnswerResponseError
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))) as http:
            with pytest.raises(AnswerResponseError):
                await make_answer(http).answer("paper-1", "question", [hit()])
    asyncio.run(run())


@pytest.mark.parametrize("status", [401, 402, 429, 500, 302])
def test_http_errors_do_not_retry_or_follow_redirects_and_do_not_echo_key(status):
    from app.rag.answer import AnswerResponseError
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(status, headers={"location": "https://other.example"}, text="unit-test-key")
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True) as http:
            with pytest.raises(AnswerResponseError) as caught:
                await make_answer(http).answer("paper-1", "question", [hit()])
            assert caught.value.status_code == status
            assert "unit-test-key" not in str(caught.value)
            assert isinstance(caught.value.__cause__, httpx.HTTPStatusError)
            assert not http.is_closed
    asyncio.run(run())
    assert len(calls) == 1


@pytest.mark.parametrize("exception,name", [(httpx.ReadTimeout, "AnswerTimeoutError"),
    (httpx.ConnectError, "AnswerConnectionError"), (httpx.RemoteProtocolError, "AnswerConnectionError")])
def test_transport_errors_preserve_cause(exception, name):
    from app.rag import answer as module
    def handler(request):
        raise exception("simulated", request=request)
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            with pytest.raises(getattr(module, name)) as caught:
                await make_answer(http).answer("paper-1", "question", [hit()])
            assert isinstance(caught.value.__cause__, exception)
    asyncio.run(run())


def test_cancellation_propagates():
    def handler(request):
        raise asyncio.CancelledError()
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            with pytest.raises(asyncio.CancelledError):
                await make_answer(http).answer("paper-1", "question", [hit()])
    asyncio.run(run())
