import asyncio
import json

import httpx
import pytest
from pydantic import SecretStr

from app.observability.store import InMemoryTraceStore
from tests.test_web_search import result


def test_singleflight_notification_capacity_release_and_waiter_limits():
    from app.tools.search_coalescing import SearchSingleFlight
    async def run():
        flights=SearchSingleFlight(max_keys=1,max_waiters=1)
        first=flights.claim('a');second=flights.claim('a')
        assert first.leader and not second.leader
        assert flights.claim('a') is None and flights.claim('b') is None
        flights.release_waiter(second)
        replacement=flights.claim('a')
        assert replacement is not None
        flights.finish(first,error='failed')
        assert (await replacement.future).error=='failed'
        flights.release_waiter(replacement)
        assert flights.claim('b').leader
    asyncio.run(run())


def test_singleflight_clear_notifies_waiters_and_old_owner_cannot_remove_new_flight():
    from app.tools.search_coalescing import SearchSingleFlight
    async def run():
        flights=SearchSingleFlight()
        first=flights.claim('a');waiter=flights.claim('a')
        flights.clear()
        assert (await waiter.future).error=='invalidated'
        current=flights.claim('a')
        flights.finish(first,error='failed')
        assert not current.future.done()
        flights.finish(current,error='failed')
        flights.release_waiter(waiter)
    asyncio.run(run())


class Harness:
    def __init__(self,*,callers=4,cache_options=None,empty=False,invalid=False,fail=False):
        from app.tools.search_cache import MemorySearchCache
        self.looked=asyncio.Event();self.network_started=asyncio.Event();self.release=asyncio.Event()
        self.network=[];self.traces=InMemoryTraceStore();self.empty=empty;self.fail=fail;self.invalid=invalid
        harness=self
        class Cache(MemorySearchCache):
            calls=0
            async def get(self,key):
                self.calls+=1
                if self.calls>=callers:harness.looked.set()
                return await super().get(key)
        self.cache=Cache(**(cache_options or {}))
    async def handler(self,request):
        self.network.append(json.loads(request.content))
        self.network_started.set()
        await self.release.wait()
        if self.fail:return httpx.Response(500,text='private-provider-error')
        return httpx.Response(200,json={'results':[] if self.empty else [result(url='javascript:x') if self.invalid else result()]})
    async def search(self,index,*,query='q',secret='secret',timeout=2):
        from app.tools.web_search import DeepSeekSearchClient
        async with httpx.AsyncClient(transport=httpx.MockTransport(self.handler)) as http:
            client=DeepSeekSearchClient(client=http,api_key=SecretStr(secret),trace_store=self.traces,
                cache=self.cache,timeout_seconds=timeout)
            return await client.search(query,request_id='r'+str(index),trace_id='t'+str(index))
    async def start(self,count=4):
        tasks=[asyncio.create_task(self.search(i)) for i in range(count)]
        await asyncio.wait_for(self.looked.wait(),1)
        return tasks


def test_four_identical_cold_calls_share_one_network_request_and_current_traces():
    async def run():
        h=Harness();tasks=await h.start();h.release.set()
        batches=await asyncio.gather(*tasks)
        assert len(h.network)==1
        assert len({b.trace_id for b in batches})==4 and len({b.retrieved_at for b in batches})==1
        traces=[await h.traces.get('t'+str(i)) for i in range(4)]
        attrs=[t.spans[0].attributes for t in traces]
        assert sum(a['outbound_request_attempts'] for a in attrs)==1
        assert sum(a['response_body_bytes']>0 for a in attrs)==1
        assert [a['coalescing_role'] for a in attrs].count('follower')==3
        assert all(a['cache_status']=='miss' for a in attrs)
        owner=next(b.trace_id for b,a in zip(batches,attrs) if a['coalescing_role']=='leader')
        assert all(a['origin_search_trace_id']==owner for a in attrs if a['coalescing_role']=='follower')
        assert 'private-provider-error' not in ''.join(t.model_dump_json() for t in traces)
        # Completed results become ordinary cache hits, not lingering in-flight entries.
        await h.search(5)
        assert (await h.traces.get('t5')).spans[0].attributes['cache_status']=='hit'
    asyncio.run(run())


@pytest.mark.parametrize('variation',['query','credential'])
def test_different_keys_do_not_share_or_serialize(variation):
    async def run():
        h=Harness(callers=2)
        first=asyncio.create_task(h.search(0))
        second=asyncio.create_task(h.search(1,query='different' if variation=='query' else 'q',
                                                   secret='other' if variation=='credential' else 'secret'))
        await asyncio.wait_for(h.looked.wait(),1)
        assert len(h.network)==2
        h.release.set();await asyncio.gather(first,second)
    asyncio.run(run())


@pytest.mark.parametrize('options',[{'coalesce_enabled':False},{'max_waiters':1},{'max_inflight':1}])
def test_disable_or_capacity_fallback_is_explicit_and_bounded(options):
    async def run():
        h=Harness(callers=3,cache_options=options)
        if 'max_inflight' in options:
            tasks=[asyncio.create_task(h.search(i,query=str(i))) for i in range(3)]
            await asyncio.wait_for(h.looked.wait(),1)
        else:tasks=await h.start(3)
        h.release.set();await asyncio.gather(*tasks)
        assert len(h.network)==(2 if 'max_waiters' in options else 3)
        statuses=[(await h.traces.get('t'+str(i))).spans[0].attributes['coalescing_status'] for i in range(3)]
        assert ('disabled' if 'coalesce_enabled' in options else 'capacity_bypass') in statuses
    asyncio.run(run())


@pytest.mark.parametrize('failure',['http','invalid','cancel'])
def test_owner_failure_or_cancellation_wakes_waiters_without_hidden_retries(failure):
    from app.tools.web_search import SearchError
    async def run():
        h=Harness(callers=2,fail=failure=='http',invalid=failure=='invalid')
        tasks=await h.start(2)
        if failure=='cancel':tasks[0].cancel()
        else:h.release.set()
        outcomes=await asyncio.wait_for(asyncio.gather(*tasks,return_exceptions=True),1)
        assert isinstance(outcomes[0],asyncio.CancelledError if failure=='cancel' else SearchError)
        assert isinstance(outcomes[1],SearchError)
        assert len(h.network)==1
        h.release.set();h.fail=False;h.invalid=False
        assert (await h.search(3)).results
        assert len(h.network)==2
    asyncio.run(run())


def test_cancelling_one_waiter_does_not_cancel_owner_or_other_waiters():
    async def run():
        h=Harness(callers=3);tasks=await h.start(3)
        tasks[1].cancel()
        with pytest.raises(asyncio.CancelledError):await tasks[1]
        assert not tasks[0].done() and not tasks[2].done()
        h.release.set();found=await asyncio.gather(tasks[0],tasks[2])
        assert all(b.results for b in found) and len(h.network)==1
    asyncio.run(run())


def test_waiter_deadline_does_not_cancel_owner():
    from app.tools.web_search import SearchTimeoutError
    async def run():
        h=Harness(callers=2)
        owner=asyncio.create_task(h.search(0))
        await asyncio.wait_for(h.network_started.wait(),1)
        with pytest.raises(SearchTimeoutError):await h.search(1,timeout=.01)
        assert not owner.done()
        h.release.set();assert (await owner).results
        assert len(h.network)==1
    asyncio.run(run())


@pytest.mark.parametrize('options',[{'empty':True},{'cache_options':{'max_entry_bytes':1}}])
def test_same_live_result_can_be_shared_even_when_not_admitted_to_ttl_cache(options):
    async def run():
        h=Harness(**options);tasks=await h.start();h.release.set()
        batches=await asyncio.gather(*tasks)
        assert len(h.network)==1 and all(b.results==batches[0].results for b in batches)
        await h.search(6)
        assert len(h.network)==2
    asyncio.run(run())


def test_clear_notifies_waiters_and_prevents_old_owner_refilling_invalidated_cache():
    from app.tools.web_search import SearchError
    async def run():
        h=Harness(callers=2);tasks=await h.start(2)
        h.cache.clear()
        with pytest.raises(SearchError):await asyncio.wait_for(tasks[1],1)
        h.release.set();assert (await tasks[0]).results
        await h.search(7)
        assert len(h.network)==2
    asyncio.run(run())
