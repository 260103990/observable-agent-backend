"""Request-owned Loop resources, independent of the original Agent dependency."""
from uuid import uuid4

from fastapi import Depends, HTTPException, Request
import httpx
from pydantic import ConfigDict

from app.agent.deepseek_loop import DeepSeekLoopModel
from app.agent.web_evidence import WEB_SYSTEM_PROMPT, GroundedWebAnswer, WebEvidenceTool
from app.tools.web_models import SearchInput
from app.tools.web_search import DeepSeekSearchClient
from app.agent.loop_models import LoopLimits
from app.agent.loop_runner import AgentLoopRunner
from app.agent.paper_tool import PAPER_SYSTEM_PROMPT, PAPER_CONTEXT_PROMPT, PaperEvidenceTool, RetrievalInput
from app.api.dependencies import get_trace_store
from app.config import Settings, get_settings
from app.schemas.agent_loop import AgentLoopRequest
from app.tools.async_registry import AsyncTool, AsyncToolRegistry
from app.tools.calculator import AddNumbersInput, add_numbers


class StrictAddNumbersInput(AddNumbersInput):
    model_config = ConfigDict(strict=True,extra='forbid')


async def get_loop_runner(request: Request, body: AgentLoopRequest,
                          settings: Settings=Depends(get_settings), trace_store=Depends(get_trace_store)):
    request.state.trace_id=str(uuid4())
    if settings.deepseek_api_key is None:
        raise HTTPException(status_code=503,detail='Agent Loop requires a locally configured DeepSeek key')
    if body.enable_web_search and not settings.web_search_enabled:
        raise HTTPException(status_code=503,detail='Web Search is not enabled on this server')
    rag=getattr(request.app.state,'rag_service',None)
    if body.paper_id is not None and rag is None:
        raise HTTPException(status_code=503,detail='Paper retrieval requires the enabled RAG service')
    limits=LoopLimits(max_final_repairs=settings.agent_loop_max_final_repairs,
        max_rounds=settings.agent_loop_max_rounds,
        max_tool_calls=settings.agent_loop_max_tool_calls,max_calls_per_turn=settings.agent_loop_max_calls_per_turn,
        model_timeout_seconds=settings.agent_loop_model_timeout_seconds,
        tool_timeout_seconds=settings.agent_loop_tool_timeout_seconds,
        run_timeout_seconds=settings.agent_loop_run_timeout_seconds,
        trace_timeout_seconds=settings.agent_loop_trace_timeout_seconds,
        max_history_chars=settings.agent_loop_max_history_chars,
        max_tool_result_chars=settings.agent_loop_max_tool_result_chars)
    registry=AsyncToolRegistry()
    async def add(arguments):
        return add_numbers(arguments)
    registry.register(AsyncTool('add_numbers','Add two integers.',StrictAddNumbersInput,add,read_only=True))
    options={}
    paper=None
    if body.paper_id is not None:
        paper=PaperEvidenceTool(rag=rag,paper_id=body.paper_id,request_id=request.state.request_id,top_k=body.top_k,
            max_chars=settings.rag_max_evidence_chars,max_items=settings.rag_max_evidence_items,
            max_result_chars=limits.max_tool_result_chars)
        registry.register(AsyncTool('paper_retrieval','Retrieve evidence from the server-selected paper. Rephrase query if needed.',
                                   RetrievalInput,paper.retrieve,read_only=True))
        options={'system_prompt':PAPER_SYSTEM_PROMPT,'finalizer':paper.finalize,
                 'evidence_history_tools':('paper_retrieval',)}
        if settings.rag_retrieval_strategy=='hybrid':
            options['system_prompt']+=PAPER_CONTEXT_PROMPT
    # One client is reused across this run's model turns; it closes on success, error or cancellation.
    async with httpx.AsyncClient(trust_env=False,limits=httpx.Limits(max_connections=4,max_keepalive_connections=4)) as http:
        if body.enable_web_search:
            search=DeepSeekSearchClient(client=http,api_key=settings.deepseek_api_key,trace_store=trace_store,
                timeout_seconds=settings.web_search_timeout_seconds,
                max_response_bytes=settings.web_search_max_response_bytes,
                trace_timeout_seconds=limits.trace_timeout_seconds,
                cache=getattr(request.app.state,'search_cache',None) if settings.web_search_cache_enabled else None,
                cache_timeout_seconds=settings.web_search_cache_timeout_seconds)
            web=WebEvidenceTool(search=search,request_id=request.state.request_id,
                max_items=settings.web_search_max_evidence_items,max_chars=settings.web_search_max_evidence_chars)
            registry.register(AsyncTool('web_search','Search the live web for external facts. Returns provider summaries, not fetched full pages.',
                                       SearchInput,web.search,read_only=True))
            options={'system_prompt':WEB_SYSTEM_PROMPT,'finalizer':GroundedWebAnswer(web,paper=paper).finalize,
                     'evidence_history_tools':('web_search','paper_retrieval') if paper else ('web_search',)}
            if paper and settings.rag_retrieval_strategy=='hybrid':
                options['system_prompt']+=PAPER_CONTEXT_PROMPT
        if options:
            options['answer_scope_policy_enabled']=settings.answer_scope_policy_enabled
        model=DeepSeekLoopModel(client=http,api_key=settings.deepseek_api_key,model=settings.deepseek_model,
            chat_url=settings.deepseek_chat_url,timeout_seconds=limits.model_timeout_seconds,
            max_tokens=settings.deepseek_max_tokens,max_calls_per_turn=limits.max_calls_per_turn)
        yield AgentLoopRunner(registry=registry,model=model,trace_store=trace_store,limits=limits,**options)
