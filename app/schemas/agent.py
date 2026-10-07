from pydantic import BaseModel, Field, field_validator


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, object]


class AgentRunRequest(BaseModel):
    prompt: str = Field(min_length=1)

    @field_validator("prompt", mode="before")
    @classmethod
    def strip_prompt(cls, value):
        if isinstance(value, str):
            return value.strip()

        return value