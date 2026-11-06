"""Additive loop API; the legacy AgentRunResponse is unchanged."""
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.agent.loop_models import LoopResult


class AgentLoopRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    enable_web_search: bool = False
    prompt: str = Field(min_length=1,max_length=4000)
    paper_id: str | None = Field(default=None,min_length=1,max_length=128)
    top_k: int = Field(default=5,ge=1,le=100)

    @field_validator('prompt','paper_id',mode='before')
    @classmethod
    def trim(cls,value):
        return value.strip() if isinstance(value,str) else value


class AgentLoopResponse(LoopResult):
    trace_id: str
