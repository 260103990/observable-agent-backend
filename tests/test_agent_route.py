import pytest
from pydantic import ValidationError


def test_agent_run_request_strips_prompt():
    from app.schemas.agent import AgentRunRequest

    request = AgentRunRequest(
        prompt="  请计算 17 + 25  ",
    )

    assert request.prompt == "请计算 17 + 25"


@pytest.mark.parametrize("prompt", ["", "   "])
def test_agent_run_request_rejects_blank_prompt(prompt):
    from app.schemas.agent import AgentRunRequest

    with pytest.raises(ValidationError):
        AgentRunRequest(prompt=prompt)