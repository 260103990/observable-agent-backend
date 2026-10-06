import httpx

from app.schemas.agent import ToolCall
from app.services.ollama_service import chat_with_tools
from pydantic import ValidationError


class OllamaDecisionError(Exception):
    pass


class OllamaDecisionMaker:
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
        tool_schemas: list[dict[str, object]],
    ) -> ToolCall:
        data = await chat_with_tools(
            model=self._model,
            prompt=prompt,
            tool_schemas=tool_schemas,
            chat_url=self._chat_url,
            timeout_seconds=self._timeout_seconds,
            client=self._client,
        )

        try:
            message = data["message"]
            tool_calls = message["tool_calls"]

            if (
                not isinstance(tool_calls, list)
                or len(tool_calls) != 1
            ):
                raise OllamaDecisionError(
                    "Ollama must return exactly one tool call"
                )

            function_call = tool_calls[0]["function"]

            return ToolCall.model_validate(
                function_call
            )

        except (
            KeyError,
            TypeError,
            ValidationError,
        ) as error:
            raise OllamaDecisionError(
                "Ollama returned an invalid tool call"
            ) from error