from fastapi import APIRouter, Depends, HTTPException

from app.config import Settings, get_settings
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.ollama_service import (
    OllamaConnectionError,
    OllamaResponseError,
    OllamaTimeoutError,
    generate,
)


router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
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
