from pydantic import BaseModel, Field, field_validator


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
