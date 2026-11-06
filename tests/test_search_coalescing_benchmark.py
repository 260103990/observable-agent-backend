import asyncio

import httpx

from app.config import Settings
from tests.test_web_search import result


def test_concurrent_benchmark_controls_cold_cache_and_counts_shared_cost_once(tmp_path):
    from app.evaluation.search_coalescing_benchmark import evaluate_search_coalescing
    requests=[];clients=[]
    async def handler(request):
        requests.append(request)
        await asyncio.sleep(0)
        return httpx.Response(200,json={'results':[result()]})
    def factory():
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        clients.append(client)
        return client
    report=asyncio.run(evaluate_search_coalescing(client_factory=factory,settings=Settings(_env_file=None,deepseek_api_key='unit-secret'),
        query='fixed',count=1,callers=4,repeats=2,output_dir=tmp_path/'run',verification_tier='controlled'))
    assert len(requests)==10 and len(clients)==16 and all(c.is_closed for c in clients)
    assert report['modes']['disabled']['search_request_attempts']==8
    assert report['modes']['enabled']['search_request_attempts']==2
    assert report['modes']['enabled']['coalesced_calls']==6
    assert all(row['cold_overlap_control_valid'] for row in report['bursts'] if row['mode']=='disabled')
    assert all(row['coalescing_exercised'] for row in report['bursts'] if row['mode']=='enabled')
    assert report['manual_review']['source_quality']=='pending'


def test_benchmark_records_provider_failures_without_retries_or_secret_bodies(tmp_path):
    from app.evaluation.search_coalescing_benchmark import evaluate_search_coalescing
    requests=[]
    async def handler(request):
        requests.append(request)
        await asyncio.sleep(0)
        return httpx.Response(500,text='unit-secret private provider response')
    def factory():return httpx.AsyncClient(transport=httpx.MockTransport(handler))
    report=asyncio.run(evaluate_search_coalescing(client_factory=factory,settings=Settings(_env_file=None,deepseek_api_key='unit-secret'),
        query='q',count=1,callers=3,repeats=1,output_dir=tmp_path/'run',verification_tier='controlled'))
    assert len(requests)==4
    assert report['modes']['disabled']['failures']==report['modes']['enabled']['failures']==3
    assert report['modes']['enabled']['coalesced_calls']==0
    assert not any(row['coalescing_exercised'] for row in report['bursts'])
    assert 'unit-secret' not in ''.join(p.read_text(encoding='utf-8') for p in (tmp_path/'run').iterdir())
