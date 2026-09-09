from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from app.services.ollama_service import (
        OllamaTimeoutError, 
        generate,
)

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
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        answer = await generate(
            model="qwen2.5:1.5b",
            prompt=request.message,
        )
    except OllamaTimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail="Ollama request timed out",
        ) from exc

    return ChatResponse(answer=answer)