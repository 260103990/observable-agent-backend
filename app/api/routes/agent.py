from fastapi import APIRouter, Depends, HTTPException

from app.agent.ollama_decision_maker import (
    OllamaDecisionError,
)
from app.agent.runner import AgentRunner
from app.api.dependencies import get_agent_runner
from app.schemas.agent import (
    AgentRunRequest,
    AgentRunResponse,
)
from app.services.ollama_service import (
    OllamaConnectionError,
    OllamaResponseError,
    OllamaTimeoutError,
)
from app.tools.registry import (
    ToolArgumentsError,
    ToolNotFoundError,
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
    try:
        result = await runner.run(request.prompt)
    except OllamaTimeoutError as error:
        raise HTTPException(
            status_code=504,
            detail="Ollama request timed out",
        ) from error
    except OllamaConnectionError as error:
        raise HTTPException(
            status_code=503,
            detail="Ollama service is unavailable",
        ) from error
    except OllamaResponseError as error:
        raise HTTPException(
            status_code=502,
            detail="Ollama returned an error response",
        ) from error
    except OllamaDecisionError as error:
        raise HTTPException(
            status_code=502,
            detail="Ollama returned an invalid tool decision",
        ) from error
    except ToolNotFoundError as error:
        raise HTTPException(
            status_code=502,
            detail="Ollama selected an unknown tool",
        ) from error
    except ToolArgumentsError as error:
        raise HTTPException(
            status_code=502,
            detail="Ollama returned invalid tool arguments",
        ) from error

    return AgentRunResponse(result=result)
