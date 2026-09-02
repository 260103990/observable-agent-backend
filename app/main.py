from fastapi import FastAPI
from pydantic import BaseModel
from app.services.ollama_service import generate

app = FastAPI(title="Observable Agent Backend")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    answer = await generate(
        model="qwen2.5:1.5b",
        prompt=request.message,
    )

    return ChatResponse(answer=answer)