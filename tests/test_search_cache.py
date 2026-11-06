import asyncio
import json

import httpx
import pytest
from pydantic import SecretStr

from app.observability.store import InMemoryTraceStore
from app.tools.web_models import SearchBatch,SearchResult
from tests.test_web_search import result


class Clock:
    now=0.0
    def __call__(self):return self.now


def batch(label='x',**changes):
    fields=dict(query=label,retrieved_at='2026-11-05T00:00:00+00:00',trace_id='origin',
        response_body_bytes=100,results=(SearchResult(rank=1,title='Title',url='https://example.org',
        snippet='Snippet',summary='Summary',rerank_score=2.0,authority=3),))
    fields.update(changes)
    return SearchBatch(**fields)


def test_key_is_exact_request_profile_and_credential_scoped():
    from app.tools.search_cache import search_cache_key
    key=search_cache_key('  Q  x  ',5,api_key=SecretStr('private-a'),max_response_bytes=1000)
    assert key==search_cache_key('Q  x',5,api_key=SecretStr('private-a'),max_response_bytes=1000)
    assert len(key)==64 and 'private-a' not in key and 'Q' not in key
    for query,count,secret,size in [('q  x',5,'private-a',1000),('Q x',5,'private-a',1000),
        ('Q  x',6,'private-a',1000),('Q  x',5,'private-b',1000),('Q  x',5,'private-a',999)]:
        assert key!=search_cache_key(query,count,api_key=SecretStr(secret),max_response_bytes=size)


def test_ttl_expires_at_boundary_and_reads_do_not_extend_it():
    from app.tools.search_cache import MemorySearchCache
    async def run():
        clock=Clock();cache=MemorySearchCache(ttl_seconds=10,clock=clock)
        assert await cache.put('a',batch())
        clock.now=9
        hit=await cache.get('a')
        assert hit.batch==batch() and hit.age_seconds==9
        clock.now=10
        assert await cache.get('a') is None
    asyncio.run(run())


def test_lru_capacity_is_bounded_and_clear_invalidates():
    from app.tools.search_cache import MemorySearchCache
    async def run():
        cache=MemorySearchCache(max_entries=2)
        for key in ('a','b'):assert await cache.put(key,batch(key))
        assert (await cache.get('a')).batch.query=='a'
        await cache.put('c',batch('c'))
        assert await cache.get('b') is None
        assert await cache.get('a') and await cache.get('c')
        cache.clear()
        assert await cache.get('a') is None and await cache.get('c') is None
    asyncio.run(run())


def test_serialized_byte_capacity_and_oversized_admission():
    from app.tools.search_cache import MemorySearchCache
    async def run():
        size=len(batch().model_dump_json().encode('utf-8'))
        cache=MemorySearchCache(max_bytes=size,max_entry_bytes=size)
        assert await cache.put('a',batch())
        assert await cache.put('b',batch())
        assert await cache.get('a') is None
        large=batch('x'*100)
        assert not await cache.put('large',large)
        assert await cache.get('large') is None and await cache.get('b')
    asyncio.run(run())


def test_cache_returns_independent_validated_snapshot():
    from app.tools.search_cache import MemorySearchCache
    async def run():
        cache=MemorySearchCache()
        value=batch()
        await cache.put('a',value)
        one=(await cache.get('a')).batch
        two=(await cache.get('a')).batch
        assert one==two==value and one is not two and one.results[0] is not two.results[0]
    asyncio.run(run())


@pytest.mark.parametrize('options',[{'ttl_seconds':0},{'ttl_seconds':True},{'ttl_seconds':float('nan')},
    {'max_entries':0},{'max_bytes':True},{'max_entry_bytes':0}])
def test_cache_rejects_invalid_resource_settings(options):
    from app.tools.search_cache import MemorySearchCache
    with pytest.raises(ValueError):MemorySearchCache(**options)


def test_search_hit_preserves_source_time_but_uses_current_trace_and_zero_network_cost():
    from app.tools.search_cache import MemorySearchCache
    from app.tools.web_search import DeepSeekSearchClient
    async def run():
        requests=[];traces=InMemoryTraceStore();cache=MemorySearchCache()
        def handler(request):
            requests.append(request)
            return httpx.Response(200,json={'results':[result()]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            for index in range(2):
                search=DeepSeekSearchClient(client=http,api_key=SecretStr('private-secret'),trace_store=traces,cache=cache)
                found=await search.search('q',count=5,request_id=f'r{index}',trace_id=f't{index}')
                if index==0:first=found
            assert len(requests)==1 and found.results==first.results
            assert found.retrieved_at==first.retrieved_at and found.trace_id=='t1' and found.response_body_bytes==0
            old=await traces.get('t0');new=await traces.get('t1')
            assert old.spans[0].attributes['cache_status']=='miss'
            attrs=new.spans[0].attributes
            assert attrs['cache_status']=='hit' and attrs['origin_search_trace_id']=='t0'
            assert attrs['outbound_request_attempts']==attrs['response_body_bytes']==0
            assert not any(s.name=='search.http' for s in new.spans)
            assert 'private-secret' not in new.model_dump_json()
    asyncio.run(run())


@pytest.mark.parametrize('response',[httpx.Response(200,json={'results':[]}),httpx.Response(500),
                                   httpx.Response(200,json={'results':[result(url='javascript:x')]} )])
def test_empty_and_failed_provider_results_are_not_cached(response):
    from app.tools.search_cache import MemorySearchCache
    from app.tools.web_search import DeepSeekSearchClient,SearchError
    async def run():
        calls=[];cache=MemorySearchCache()
        def handler(request):calls.append(request);return response
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            search=DeepSeekSearchClient(client=http,api_key=SecretStr('secret'),trace_store=InMemoryTraceStore(),cache=cache)
            for index in range(2):
                try:await search.search('q',request_id='r',trace_id=str(index))
                except SearchError:pass
        assert len(calls)==2
    asyncio.run(run())


@pytest.mark.parametrize('operation',['get','put'])
@pytest.mark.parametrize('failure',['error','timeout'])
def test_cache_failure_falls_back_or_keeps_success_without_retry(operation,failure):
    from app.tools.web_search import DeepSeekSearchClient
    class Broken:
        async def get(self,key):
            if operation=='get':await self.fail()
            return None
        async def put(self,*args):
            if operation=='put':await self.fail()
            return True
        async def fail(self):
            if failure=='timeout':await asyncio.Event().wait()
            raise RuntimeError('private-cache-error')
    async def run():
        calls=[];traces=InMemoryTraceStore()
        def handler(request):calls.append(request);return httpx.Response(200,json={'results':[result()]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            search=DeepSeekSearchClient(client=http,api_key=SecretStr('secret'),trace_store=traces,
                cache=Broken(),cache_timeout_seconds=.01)
            found=await search.search('q',request_id='r',trace_id='t')
        assert found.results and len(calls)==1
        trace=await traces.get('t')
        assert trace.status=='ok' and 'private-cache-error' not in trace.model_dump_json()
        assert trace.spans[0].attributes['cache_error_type'] in ('RuntimeError','TimeoutError')
    asyncio.run(run())


def test_cancellation_during_cache_lookup_propagates_without_network():
    from app.tools.web_search import DeepSeekSearchClient
    async def run():
        calls=[];entered=asyncio.Event();traces=InMemoryTraceStore()
        class Waiting:
            async def get(self,key):entered.set();await asyncio.Event().wait()
        def handler(request):calls.append(request);return httpx.Response(200,json={'results':[]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            search=DeepSeekSearchClient(client=http,api_key=SecretStr('secret'),trace_store=traces,
                cache=Waiting(),cache_timeout_seconds=10)
            task=asyncio.create_task(search.search('q',request_id='r',trace_id='t'))
            await asyncio.wait_for(entered.wait(),1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):await task
        assert not calls and (await traces.get('t')).status=='cancelled'
    asyncio.run(run())
