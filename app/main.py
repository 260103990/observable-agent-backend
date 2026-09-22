from time import perf_counter
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from app.services.ollama_service import (
        OllamaTimeoutError, 
        OllamaConnectionError,
        OllamaResponseError,
        generate,
)
from app.config import Settings, get_settings

app = FastAPI(title="Observable Agent Backend")
@app.middleware("http")
async def add_request_context(
    request: Request,
    call_next,
):
    start_time = perf_counter()

    request_id = (
        request.headers.get("X-Request-ID")
        or str(uuid4())
    )

    response = await call_next(request)

    process_time_ms = (
        perf_counter() - start_time
    ) * 1000

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = (
        f"{process_time_ms:.2f}"
    )

    return response

class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    @field_validator("message", mode="before")
    @classmethod
    def strip_message(cls, value):
        if isinstance(value, str):
            return value.strip()
        return value

class ChatResponse(BaseModel):
    answer: str


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    settings: Settings = Depends(get_settings),
) -> ChatResponse:
    try:
        answer = await generate(
            model=settings.ollama_model,
            prompt=request.message,
            generate_url=settings.ollama_generate_url,
            timeout_seconds=settings.ollama_timeout_seconds,
        )
    except OllamaTimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail="Ollama request timed out",
        ) from exc
    except OllamaConnectionError as exc:
        raise HTTPException(
            status_code=503,
            detail="Ollama service is unavailable",
        ) from exc
    except OllamaResponseError as exc:
        raise HTTPException(
            status_code=502,
            detail="Ollama returned an error response",
        ) from exc
    return ChatResponse(answer=answer)