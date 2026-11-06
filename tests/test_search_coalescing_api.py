import asyncio
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException

from app.config import Settings,get_settings
from app.main import app
from app.api.dependencies import get_trace_store
from app.observability.store import InMemoryTraceStore
from tests.test_agent_loop_api import message,call
from tests.test_web_loop_api import search_response,final


def test_coalescing_settings_and_memory_factory():
    from app.tools.search_cache import cache_from_settings
    settings=Settings(_env_file=None)
    assert settings.web_search_cache_coalesce_enabled is True
    assert settings.web_search_cache_max_inflight==64 and settings.web_search_cache_max_waiters==32
    for options in ({'web_search_cache_max_inflight':0},{'web_search_cache_max_waiters':0}):
        with pytest.raises(ValueError):Settings(_env_file=None,**options)
    cache=cache_from_settings(settings.model_copy(update={'web_search_cache_coalesce_enabled':False}))
    assert cache.flights.enabled is False


def test_trace_metrics_separate_waiting_from_cached_hits():
    from app.evaluation.scoring import summarize_traces
    from tests.test_agent_evaluation import trace
    owner=trace('owner','web.search',cache_status='miss',coalescing_role='leader',outbound_request_attempts=1,response_body_bytes=100)
    follower=trace('follower','web.search',cache_status='miss',coalescing_role='follower',coalescing_status='reused',
                   outbound_request_attempts=0,response_body_bytes=0)
    cached=trace('cached','web.search',cache_status='hit',outbound_request_attempts=0,response_body_bytes=0)
    metrics=summarize_traces([owner,follower,cached],root_trace_id='missing')
    assert metrics['search_coalesced_calls']==1 and metrics['search_coalescing_waits']==1
    assert metrics['search_cache_hits']==1 and metrics['search_cache_misses']==2
    assert metrics['search_request_attempts']==1 and metrics['search_response_body_bytes']==100


def test_concurrent_asgi_requests_keep_separate_evidence_and_close_owned_clients(monkeypatch):
    from app import main
    from app.api import loop_dependencies as deps
    from app.tools.search_cache import MemorySearchCache
    async def run():
        all_lookups=asyncio.Event()
        class Cache(MemorySearchCache):
            lookups=0
            async def get(self,key):
                self.lookups+=1
                if self.lookups>=3:all_lookups.set()
                return await super().get(key)
        cache=Cache();traces=InMemoryTraceStore();clients=[];searches=[]
        async def external(request):
            if str(request.url).endswith('/tools/search'):
                searches.append(request)
                await asyncio.wait_for(all_lookups.wait(),1)
                return httpx.Response(200,json=search_response())
            import json
            data=json.loads(request.content)
            if data['messages'][-1]['role']=='tool':return httpx.Response(200,json=final())
            return httpx.Response(200,json=message(calls=[call('tool','web_search',{'query':'same query','count':1})]))
        original=httpx.AsyncClient
        def factory(**kwargs):
            client=original(transport=httpx.MockTransport(external),**kwargs)
            clients.append(client)
            return client
        settings=Settings(_env_file=None,deepseek_api_key='unit-secret',web_search_enabled=True,web_search_cache_enabled=True)
        monkeypatch.setattr(main,'get_settings',lambda:settings)
        monkeypatch.setattr(main,'cache_from_settings',lambda settings:cache)
        monkeypatch.setattr(deps,'httpx',SimpleNamespace(AsyncClient=factory,Limits=httpx.Limits))
        app.dependency_overrides[get_settings]=lambda:settings
        app.dependency_overrides[get_trace_store]=lambda:traces
        try:
            async with app.router.lifespan_context(app):
                async with original(transport=httpx.ASGITransport(app=app),base_url='http://test') as http:
                    responses=await asyncio.gather(*(http.post('/agent/loop',json={'prompt':'q','enable_web_search':True},
                        headers={'X-Request-ID':str(i)}) for i in range(3)))
                assert all(r.status_code==200 for r in responses),[r.text for r in responses]
                ids=[r.json()['web_citations'][0]['search_trace_id'] for r in responses]
                assert len(set(ids))==3 and len(searches)==1
                search_traces=[await traces.get(identifier) for identifier in ids]
                assert sum(t.spans[0].attributes.get('coalescing_status')=='reused' for t in search_traces)==2
                assert all(client.is_closed for client in clients)
            assert not hasattr(app.state,'search_cache')
        finally:
            app.dependency_overrides.clear()
    asyncio.run(run())
