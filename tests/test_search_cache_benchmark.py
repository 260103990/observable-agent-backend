import asyncio
import json

import httpx
import pytest

from app.config import Settings
from tests.test_web_search import result


def workload():
    from app.evaluation.search_cache_benchmark import SearchBenchmarkWorkload
    return SearchBenchmarkWorkload.model_validate({'version':'test-v1','cases':[
        {'id':'a','query':'query a','count':1},{'id':'b','query':'query b','count':1}]})


def test_benchmark_has_distinct_no_cache_cold_priming_and_warm_phases(tmp_path):
    from app.evaluation.search_cache_benchmark import evaluate_search_cache
    requests=[]
    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200,json={'results':[result()]})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await evaluate_search_cache(client=http,settings=Settings(_env_file=None,deepseek_api_key='private-secret'),
                workload=workload(),repeats=2,output_dir=tmp_path/'run',verification_tier='controlled')
    report=asyncio.run(run())
    phases=report['phases']
    assert len(requests)==10
    assert [phases[p]['search_request_attempts'] for p in ('bypass','cold','priming','warm')]==[4,4,2,0]
    assert phases['warm']['cache_hits']==4 and phases['warm']['response_body_bytes']==0
    assert phases['warm']['automatic_reuse_pass'] is True
    assert report['manual_review']['source_quality']=='pending' and report['verification_tier']=='controlled'
    rows=[json.loads(x) for x in (tmp_path/'run/cases.jsonl').read_text(encoding='utf-8').splitlines()]
    primed={r['case_id']:r for r in rows if r['phase']=='priming'}
    for row in [r for r in rows if r['phase']=='warm']:
        origin=primed[row['case_id']]
        assert row['payload_sha256']==origin['payload_sha256'] and row['retrieved_at']==origin['retrieved_at']
        assert row['trace_id']!=origin['trace_id'] and row['origin_search_trace_id']==origin['trace_id']
    assert 'private-secret' not in ''.join(p.read_text(encoding='utf-8') for p in (tmp_path/'run').iterdir())


def test_benchmark_keeps_failed_requests_and_does_not_claim_all_success(tmp_path):
    from app.evaluation.search_cache_benchmark import evaluate_search_cache
    requests=[]
    def handler(request):
        requests.append(request)
        return httpx.Response(500,text='private-secret') if len(requests)==1 else httpx.Response(200,json={'results':[result()]})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await evaluate_search_cache(client=http,settings=Settings(_env_file=None,deepseek_api_key='fake'),
                workload=workload(),repeats=1,output_dir=tmp_path/'run',verification_tier='controlled')
    report=asyncio.run(run())
    assert report['phases']['bypass']['failures']==1
    assert report['phases']['bypass']['search_request_attempts']==2
    assert len((tmp_path/'run/cases.jsonl').read_text(encoding='utf-8').splitlines())==8
    assert 'private-secret' not in (tmp_path/'run/cases.jsonl').read_text(encoding='utf-8')


def test_benchmark_refuses_existing_output_before_network(tmp_path):
    from app.evaluation.search_cache_benchmark import evaluate_search_cache
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request:pytest.fail('No network allowed'))) as http:
            with pytest.raises(FileExistsError):
                await evaluate_search_cache(client=http,settings=Settings(_env_file=None,deepseek_api_key='fake'),
                    workload=workload(),repeats=1,output_dir=tmp_path,verification_tier='controlled')
    asyncio.run(run())


def test_benchmark_workload_rejects_duplicate_parameter_profiles_and_ids():
    from app.evaluation.search_cache_benchmark import SearchBenchmarkWorkload
    for cases in ([{'id':'a','query':'q'},{'id':'a','query':'different'}],
                  [{'id':'a','query':'q'},{'id':'b','query':' q '}],
                  [{'id':'bad','query':'q','count':0}]):
        with pytest.raises(ValueError):SearchBenchmarkWorkload.model_validate({'version':'v1','cases':cases})
