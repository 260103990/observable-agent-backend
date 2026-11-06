"""Isolated CLI runtime: read-only paper data, local traces, restored dependencies."""
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import asyncpg
import httpx

from app.api import loop_dependencies
from app.agent.deepseek_loop import DeepSeekLoopModel
from app.api.dependencies import get_trace_store
from app.config import get_settings
from app.answer_policy import ANSWER_SCOPE_POLICY_VERSION
from app.evaluation.recording import EvaluationCapture, EvaluationRecorder
from app.main import app
from app.rag.answer import DeepSeekAnswerClient
from app.rag.embedding import OllamaEmbeddingClient
from app.rag.service import RagService
from app.rag.store import PostgresPaperStore
from app.tools.search_cache import cache_from_settings


@dataclass
class EvaluationSession:
    client: object
    recorder: EvaluationRecorder
    capture: EvaluationCapture
    paper_info: object
    manifest: dict
    search_cache: object | None=None


@asynccontextmanager
async def evaluation_session(*,settings,workload,paper_id=None,database_url=None):
    needs_paper=any(case.use_paper for case in workload.cases)
    if settings.deepseek_api_key is None:
        raise ValueError('Configure DEEPSEEK_API_KEY locally for a live evaluation')
    if needs_paper and (not paper_id or not database_url):
        raise ValueError('Paper evaluation requires explicit paper_id and EVAL_DATABASE_URL')
    if app.dependency_overrides or any(hasattr(app.state,name) for name in ('rag_service','db_pool','trace_store','search_cache')):
        raise RuntimeError('Evaluation requires a fresh process, not an already running application')
    settings=settings.model_copy(update={'database_url':None,'rag_enabled':needs_paper,
                                         'web_search_enabled':any(case.enable_web_search for case in workload.cases)})
    recorder=EvaluationRecorder()
    capture=EvaluationCapture(model_url=settings.deepseek_chat_url)
    class ObservedLoopModel(DeepSeekLoopModel):
        async def __call__(self,messages,tools):
            decision=await super().__call__(messages,tools)
            capture.record_decision(decision)
            return decision
    info,service=None,None
    original_http=httpx.AsyncClient
    def observed_http(**kwargs):
        return original_http(event_hooks={'request':[capture.on_request],'response':[capture.on_response]},**kwargs)
    proxy=SimpleNamespace(AsyncClient=observed_http,Limits=httpx.Limits)
    async with AsyncExitStack() as resources:
        if needs_paper:
            pool=await asyncpg.create_pool(dsn=database_url,min_size=1,max_size=2,command_timeout=30,
                                           server_settings={'default_transaction_read_only':'on'})
            resources.push_async_callback(pool.close)
            store=PostgresPaperStore(pool)
            info=await store.get_info(paper_id)
            if info is None or info.source_sha256!=workload.paper_sha256:
                raise ValueError('Selected paper is absent or does not match the workload PDF hash')
            http=await resources.enter_async_context(observed_http(trust_env=False))
            service=RagService(store=store,embedding=OllamaEmbeddingClient.from_settings(settings,client=http),
                answer=DeepSeekAnswerClient.from_settings(settings,client=http),trace_store=recorder,settings=settings)
        manifest={'verification_tier':'live','api_transport':'in_process_asgi',
            'database_mode':'read_only_existing_paper' if needs_paper else 'none',
            'configured_generation_model':settings.deepseek_model,
            'generation_max_tokens':settings.deepseek_max_tokens,
            'configured_embedding_model':settings.ollama_embedding_model,
            'embedding_dimensions':settings.ollama_embedding_dimensions,
            'paper_evidence_budget':{'max_items':settings.rag_max_evidence_items,'max_chars':settings.rag_max_evidence_chars},
            'loop_limits':{key:getattr(settings,'agent_loop_'+key) for key in (
                'max_final_repairs','max_rounds','max_tool_calls','max_calls_per_turn','model_timeout_seconds',
                'tool_timeout_seconds','run_timeout_seconds','max_history_chars','max_tool_result_chars')},
            'search_settings':{key:getattr(settings,'web_search_'+key) for key in (
                'enabled','timeout_seconds','max_response_bytes','max_evidence_items','max_evidence_chars')},
            'search_cache':{key:getattr(settings,'web_search_cache_'+key) for key in (
                'enabled','ttl_seconds','max_entries','max_bytes','max_entry_bytes','timeout_seconds',
                'coalesce_enabled','max_inflight','max_waiters')},
            'paper':info.model_dump(exclude={'source_path'}) if info else None,
            'rag_retrieval_strategy':settings.rag_retrieval_strategy,
            'grounded_answer_policy_version':ANSWER_SCOPE_POLICY_VERSION,
            'grounded_answer_policy_enabled':settings.answer_scope_policy_enabled,
            'provider_revision_note':'Configured generation label is not a pinned remote model weight revision.'}
        app.dependency_overrides[get_settings]=lambda:settings
        app.dependency_overrides[get_trace_store]=lambda:recorder
        if service is not None:
            app.state.rag_service=service
        cache=cache_from_settings(settings) if settings.web_search_cache_enabled else None
        if cache is not None:
            app.state.search_cache=cache
        try:
            with patch.object(loop_dependencies,'httpx',proxy), patch.object(loop_dependencies,'DeepSeekLoopModel',ObservedLoopModel):
                async with original_http(transport=httpx.ASGITransport(app=app),base_url='http://eval.local',
                                         trust_env=False,timeout=None) as client:
                    yield EvaluationSession(client,recorder,capture,info,manifest,search_cache=cache)
        finally:
            if cache is not None:
                cache.clear()
                if getattr(app.state,'search_cache',None) is cache:
                    del app.state.search_cache
            app.dependency_overrides.pop(get_settings,None)
            app.dependency_overrides.pop(get_trace_store,None)
            if service is not None and getattr(app.state,'rag_service',None) is service:
                del app.state.rag_service
