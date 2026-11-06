"""Validated Ollama embeddings, with HTTP lifetime owned by the caller."""

import math
from typing import Self

import httpx

from app.config import DEFAULT_EMBEDDING_QUERY_INSTRUCTION, Settings


class EmbeddingError(Exception):
    """An embedding operation failed; no partial result is returned."""


class EmbeddingTimeoutError(EmbeddingError):
    pass


class EmbeddingConnectionError(EmbeddingError):
    pass


class EmbeddingResponseError(EmbeddingError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class OllamaEmbeddingClient:
    """Sequential batches preserve order without unbounded GPU concurrency.

    Borrowed AsyncClient is never closed here. Timeout applies to each request;
    an overall import deadline is the responsibility of the ingestion service.
    """

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        model: str = "qwen3-embedding:0.6b",
        embed_url: str = "http://127.0.0.1:11434/api/embed",
        dimensions: int = 1024,
        batch_size: int = 16,
        timeout_seconds: float = 120.0,
        query_instruction: str = DEFAULT_EMBEDDING_QUERY_INSTRUCTION,
    ) -> None:
        if type(dimensions) is not int or dimensions <= 0:
            raise ValueError("dimensions must be a positive integer")
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive and finite")
        for name, value in (("model", model), ("embed_url", embed_url),
                            ("query_instruction", query_instruction)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonblank string")

        self._client = client
        self._model = model
        self._embed_url = embed_url
        self._dimensions = dimensions
        self._batch_size = batch_size
        self._timeout_seconds = timeout_seconds
        self._query_instruction = query_instruction

    @classmethod
    def from_settings(cls, settings: Settings, *, client: httpx.AsyncClient) -> Self:
        return cls(
            client=client,
            model=settings.ollama_embedding_model,
            embed_url=settings.ollama_embed_url,
            dimensions=settings.ollama_embedding_dimensions,
            batch_size=settings.ollama_embedding_batch_size,
            timeout_seconds=settings.ollama_embedding_timeout_seconds,
            query_instruction=settings.embedding_query_instruction,
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def query_instruction(self) -> str:
        return self._query_instruction

    async def get_revision(self) -> str:
        url = httpx.URL(self._embed_url).copy_with(path="/api/tags", query=None)
        try:
            response = await self._client.get(url, timeout=self._timeout_seconds, follow_redirects=False)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise EmbeddingTimeoutError("Ollama metadata request timed out") from exc
        except httpx.RequestError as exc:
            raise EmbeddingConnectionError("Ollama metadata transport failed") from exc
        except httpx.HTTPStatusError as exc:
            raise EmbeddingResponseError("Ollama metadata returned an HTTP error", status_code=exc.response.status_code) from exc
        try:
            data = response.json()
            models = data["models"]
            matches = [m["digest"] for m in models if m["name"] == self._model]
            if len(matches) != 1 or not isinstance(matches[0], str) or not matches[0].strip():
                raise ValueError("Model revision is unavailable")
            return matches[0]
        except (ValueError, KeyError, TypeError) as exc:
            raise EmbeddingResponseError("Ollama metadata contains no valid model revision") from exc

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not isinstance(texts, list) or any(
            not isinstance(text, str) or not text.strip() for text in texts
        ):
            raise ValueError("documents must be a list of nonblank strings")
        # Retain an input snapshot across awaits and validate all inputs first.
        texts = texts.copy()
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(await self._embed_batch(texts[start:start + self._batch_size]))
        return vectors

    async def embed_query(self, query: str) -> list[float]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonblank string")
        text = f"Instruct: {self._query_instruction}\nQuery:{query}"
        return (await self._embed_batch([text]))[0]

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            response = await self._client.post(
                self._embed_url,
                json={
                    "model": self._model,
                    "input": texts,
                    "truncate": False,
                    "dimensions": self._dimensions,
                },
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise EmbeddingTimeoutError("Ollama embedding request timed out") from exc
        except httpx.RequestError as exc:
            raise EmbeddingConnectionError("Ollama embedding transport failed") from exc
        except httpx.HTTPStatusError as exc:
            raise EmbeddingResponseError(
                "Ollama embedding returned an HTTP error",
                status_code=exc.response.status_code,
            ) from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise EmbeddingResponseError("Ollama embedding returned invalid JSON") from exc

        if not isinstance(data, dict) or data.get("model") != self._model:
            raise EmbeddingResponseError("Ollama embedding returned an invalid model identity")
        values = data.get("embeddings")
        if not isinstance(values, list) or len(values) != len(texts):
            raise EmbeddingResponseError("Ollama embedding count does not match the input")

        vectors: list[list[float]] = []
        for value in values:
            if not isinstance(value, list) or len(value) != self._dimensions:
                raise EmbeddingResponseError("Ollama embedding dimension does not match configuration")
            if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in value):
                raise EmbeddingResponseError("Ollama embedding contains nonnumeric values")
            try:
                vector = [float(x) for x in value]
            except OverflowError as exc:
                raise EmbeddingResponseError("Ollama embedding contains out-of-range values") from exc
            if not all(math.isfinite(x) for x in vector) or not any(vector):
                raise EmbeddingResponseError("Ollama embedding must be finite and nonzero")
            vectors.append(vector)
        return vectors
