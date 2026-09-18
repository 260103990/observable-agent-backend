from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field, field_validator
from app.services.ollama_service import (
        OllamaTimeoutError, 
        OllamaConnectionError,
        generate,
)
from app.config import Settings, get_settings

app = FastAPI(title="Observable Agent Backend")


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
    return ChatResponse(answer=answer)