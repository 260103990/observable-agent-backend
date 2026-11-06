from fastapi import APIRouter, Depends, HTTPException, Request

from app.agent.loop_errors import (LoopConfigurationError, LoopConnectionError, LoopContextLimitError,
                                   LoopProtocolError, LoopTimeoutError)
from app.agent.loop_runner import AgentLoopRunner
from app.api.loop_dependencies import get_loop_runner
from app.rag.evidence import CitationValidationError
from app.schemas.agent_loop import AgentLoopRequest, AgentLoopResponse


router=APIRouter(tags=['agent'])


@router.post('/agent/loop',response_model=AgentLoopResponse)
async def run_agent_loop(body: AgentLoopRequest, request: Request,
                         runner: AgentLoopRunner=Depends(get_loop_runner)):
    try:
        result=await runner.run(body.prompt,request_id=request.state.request_id,trace_id=request.state.trace_id)
    except LoopTimeoutError as exc:
        raise HTTPException(status_code=504,detail='Agent Loop timed out') from exc
    except (LoopConfigurationError,LoopConnectionError) as exc:
        raise HTTPException(status_code=503,detail='Agent Loop model service is unavailable') from exc
    except LoopContextLimitError as exc:
        raise HTTPException(status_code=413,detail='Agent Loop exceeded its context budget') from exc
    except (LoopProtocolError,CitationValidationError) as exc:
        raise HTTPException(status_code=502,detail='Agent Loop returned an invalid decision or citation') from exc
    return AgentLoopResponse(**result.model_dump(),trace_id=request.state.trace_id)
