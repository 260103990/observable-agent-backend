import asyncio

from fastapi.testclient import TestClient
import pytest

from app.config import Settings,get_settings
from app.main import app
from tests.test_agent_loop_api import environment,message,call
from tests.test_web_loop_api import search_response,final


def test_cache_config_defaults_and_bounds():
    settings=Settings(_env_file=None)
    assert settings.web_search_cache_enabled is False
    assert settings.web_search_cache_ttl_seconds==300
    assert settings.web_search_cache_max_entries==128
    assert settings.web_search_cache_max_bytes==8*1024*1024
    assert settings.web_search_cache_max_entry_bytes==262144
    assert settings.web_search_cache_timeout_seconds==.05
    for values in ({'web_search_cache_ttl_seconds':0},{'web_search_cache_ttl_seconds':float('inf')},
        {'web_search_cache_max_entries':0},{'web_search_cache_max_bytes':0},
        {'web_search_cache_max_entry_bytes':0},{'web_search_cache_timeout_seconds':0}):
        with pytest.raises(ValueError):Settings(_env_file=None,**values)


def test_cross_http_request_reuse_keeps_request_scoped_ledger_and_closes_cache(environment,monkeypatch):
    from app import main
    e=environment
    e.settings.web_search_enabled=True
    e.settings.web_search_cache_enabled=True
    monkeypatch.setattr(main,'get_settings',lambda:e.settings)
    e.answers.extend([message(calls=[call('w1','web_search',{'query':'fixed query','count':1})]),search_response(),final(),
        message(calls=[call('w2','web_search',{'query':'fixed query','count':1})]),final(),final(),final()])
    with TestClient(app) as http:
        assert hasattr(app.state,'search_cache')
        cache=app.state.search_cache
        bodies=[]
        for request_id in ('first','second'):
            response=http.post('/agent/loop',json={'prompt':'q','enable_web_search':True},headers={'X-Request-ID':request_id})
            assert response.status_code==200,response.text
            bodies.append(response.json())
        search_traces=[http.get('/traces/'+b['web_citations'][0]['search_trace_id']).json() for b in bodies]
        assert [t['spans'][0]['attributes']['cache_status'] for t in search_traces]==['miss','hit']
        assert search_traces[1]['request_id']=='second'
        assert bodies[0]['web_citations'][0]['retrieved_at']==bodies[1]['web_citations'][0]['retrieved_at']
        assert bodies[0]['web_citations'][0]['search_trace_id']!=bodies[1]['web_citations'][0]['search_trace_id']
        # Cached Search data never authorizes W1 before this request has actually called its tool.
        response=http.post('/agent/loop',json={'prompt':'q','enable_web_search':True})
        assert response.status_code==502
        key=search_traces[0]['spans'][0]['attributes']['cache_key_hash']
    assert not hasattr(app.state,'search_cache')
    assert asyncio.run(cache.get(key)) is None
    assert len([r for r in e.requests if 'query' in r])==1
    assert all(client.is_closed for client in e.clients)


def test_cache_lifespan_failure_clears_owned_state(monkeypatch):
    from app import main
    from app.rag.service_errors import RagUnavailableError
    settings=Settings(_env_file=None,web_search_cache_enabled=True,rag_enabled=True,database_url=None)
    monkeypatch.setattr(main,'get_settings',lambda:settings)
    with pytest.raises(RagUnavailableError):
        with TestClient(app):pytest.fail('RAG startup must reject missing DB')
    assert not hasattr(app.state,'search_cache')


def test_metrics_keep_hit_miss_bypass_and_cache_failures_distinct():
    from app.evaluation.scoring import summarize_traces
    from tests.test_agent_evaluation import trace
    searches=[trace('s'+str(i),'web.search',cache_status=status,outbound_request_attempts=int(status!='hit'),
                    response_body_bytes=0 if status=='hit' else 100) for i,status in enumerate(('hit','miss','bypass','error'))]
    searches[-1].spans[0].attributes['cache_error_type']='TimeoutError'
    metrics=summarize_traces(searches,root_trace_id='missing')
    assert metrics['search_cache_hits']==1 and metrics['search_cache_misses']==1
    assert metrics['search_cache_bypasses']==1 and metrics['search_cache_errors']==1
    assert metrics['search_request_attempts']==3 and metrics['search_response_body_bytes']==300


def test_evaluation_runtime_owns_cache_and_restores_state_on_failure():
    from app.evaluation.runtime import evaluation_session
    from app.evaluation.models import Workload
    from tests.test_agent_evaluation import workload_dict
    async def run():
        settings=Settings(_env_file=None,deepseek_api_key='test',web_search_cache_enabled=True)
        with pytest.raises(RuntimeError,match='stop'):
            async with evaluation_session(settings=settings,workload=Workload.model_validate(workload_dict())) as session:
                assert hasattr(app.state,'search_cache')
                assert session.manifest['search_cache']['enabled'] is True
                raise RuntimeError('stop')
        assert not hasattr(app.state,'search_cache') and not app.dependency_overrides
    asyncio.run(run())
