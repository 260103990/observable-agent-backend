import json

import httpx

from app.schemas.agent import ToolDecision
from app.services.ollama_service import chat_for_summary


class OllamaSummaryError(Exception):
    pass


def parse_summary_answer(data: object) -> str:
    if not isinstance(data, dict):
        raise OllamaSummaryError(
            "Ollama returned an invalid summary response"
        )

    message = data.get("message")

    if not isinstance(message, dict):
        raise OllamaSummaryError(
            "Ollama returned an invalid summary message"
        )

    if message.get("tool_calls"):
        raise OllamaSummaryError(
            "Ollama returned tool calls instead of a final answer"
        )

    content = message.get("content")

    if not isinstance(content, str):
        raise OllamaSummaryError(
            "Ollama summary content must be a string"
        )

    if not content.strip():
        raise OllamaSummaryError(
            "Ollama returned an empty summary"
        )

    return content


class OllamaSummarizer:
    def __init__(
        self,
        *,
        model: str,
        chat_url: str,
        timeout_seconds: float,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._model = model
        self._chat_url = chat_url
        self._timeout_seconds = timeout_seconds
        self._client = client

    async def __call__(
        self,
        prompt: str,
        decision: ToolDecision,
        tool_result: object,
    ) -> str:
        try:
            data = await chat_for_summary(
                model=self._model,
                prompt=prompt,
                assistant_message=decision.assistant_message,
                tool_name=decision.tool_call.name,
                tool_result=tool_result,
                chat_url=self._chat_url,
                timeout_seconds=self._timeout_seconds,
                client=self._client,
            )
        except json.JSONDecodeError as exc:
            raise OllamaSummaryError(
                "Ollama returned an invalid summary response"
            ) from exc

        return parse_summary_answer(data)
