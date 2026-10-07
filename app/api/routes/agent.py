from fastapi import APIRouter, Depends

from app.agent.runner import AgentRunner
from app.api.dependencies import get_agent_runner
from app.schemas.agent import (
    AgentRunRequest,
    AgentRunResponse,
)


router = APIRouter()


@router.post(
    "/agent/run",
    response_model=AgentRunResponse,
)
async def run_agent(
    request: AgentRunRequest,
    runner: AgentRunner = Depends(get_agent_runner),
) -> AgentRunResponse:
    result = await runner.run(request.prompt)

    return AgentRunResponse(result=result)