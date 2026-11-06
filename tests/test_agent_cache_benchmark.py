import asyncio,json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest

from app.config import Settings
from app.evaluation.models import Workload
from app.evaluation.recording import EvaluationCapture,EvaluationRecorder
from app.observability.models import SpanRecord
from tests.test_agent_evaluation import trace,workload_dict


def workload():
    data=workload_dict()
    data['cases']=[{'id':'web','prompt':'Use fixed query','enable_web_search':True,'required_tools':['web_search']}]
    return Workload.model_validate(data)


def test_runtime_exposes_owned_cache_and_runner_cold_reset_hook(tmp_path):
    from app.evaluation.runtime import evaluation_session
    from app.evaluation.runner import evaluate_cases
    async def run():
        async with evaluation_session(settings=Settings(_env_file=None,deepseek_api_key='test',web_search_cache_enabled=True),
                workload=Workload.model_validate(workload_dict())) as session:
            assert session.search_cache is not None
            cache=session.search_cache
        resets=[]
        async def handler(request):return httpx.Response(503)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler),base_url='http://test') as http:
            await evaluate_cases(client=http,workload=workload(),recorder=EvaluationRecorder(),
                capture=EvaluationCapture(model_url='http://model'),paper_id=None,output_dir=tmp_path/'cases',manifest={},
                repeats=2,before_case=lambda:resets.append('clear'))
        assert resets==['clear','clear']
    asyncio.run(run())


class FakeCache:
    def __init__(self):self.resets=0;self.profile=None
    def clear(self):self.resets+=1;self.profile=None


def fake_sessions(*,changed_query=False,changed_source=False,phase_source_change=False,failed_search_first=False,fail=False):
    opened=[]
    @asynccontextmanager
    async def factory(*,settings,workload,paper_id=None,database_url=None):
        cache=FakeCache() if settings.web_search_cache_enabled else None
        recorder=EvaluationRecorder();capture=EvaluationCapture(model_url='http://model');calls=0
        async def handler(request):
            nonlocal calls
            calls+=1
            request_id=request.headers['x-request-id']
            root=trace(request_id+'-root',rounds=2,tool_calls=1)
            root.request_id=request_id
            root.spans.append(SpanRecord(span_id='tool',parent_span_id='root',name='tool.execute',status='ok',
                duration_ms=10,attributes={'tool_name':'web_search'},error_type=None))
            extra_failure=failed_search_first and cache is None
            if extra_failure:
                root.spans[0].attributes['tool_calls']=2
                root.spans.append(SpanRecord(span_id='failed-tool',parent_span_id='root',name='tool.execute',status='error',
                    duration_ms=10,attributes={'tool_name':'web_search'},error_type='SearchResponseError'))
            await recorder.save(root)
            query='changed' if changed_query and calls>1 else 'fixed'
            status='bypass' if cache is None else ('hit' if cache.profile==query else 'miss')
            if cache is not None:cache.profile=query
            search=trace(request_id+'-search','web.search',cache_status=status,
                outbound_request_attempts=int(status!='hit'),response_body_bytes=0 if status=='hit' else 100)
            if status=='hit':search.spans[0].attributes['cache_age_seconds']=.1
            search.request_id=request_id
            await recorder.save(search)
            snapshot={'evidence_id':'W1','query':query,'title':'Title','url':'https://example.org',
                'snippet':'Snippet','summary':'Support','retrieved_at':'2026-11-05T00:00:00Z',
                'search_trace_id':request_id+'-search'}
            if changed_source and calls>1:snapshot['summary']='Different source snapshot'
            if phase_source_change and cache is not None:snapshot['summary']='Provider changed content after bypass phase'
            with capture.for_request(request_id):
                tool={'role':'tool','tool_call_id':'c1','content':json.dumps({'ok':True,'output':{'evidence':[snapshot]},'attempts':1})}
                assistant={'role':'assistant','content':None,'tool_calls':[{'id':'c1','type':'function',
                    'function':{'name':'web_search','arguments':json.dumps({'query':query,'count':3})}}]}
                messages=[assistant,tool]
                if extra_failure:
                    messages=[{'role':'assistant','content':None,'tool_calls':[{'id':'c0','type':'function',
                        'function':{'name':'web_search','arguments':'{"query":"different failed query","count":3}'}}]},
                        {'role':'tool','tool_call_id':'c0','content':json.dumps({'ok':False,'error':'tool_failed','attempts':1})},*messages]
                await capture.on_request(httpx.Request('POST','http://model',json={'messages':messages}))
            if fail:return httpx.Response(502,headers={'X-Trace-ID':root.trace_id})
            return httpx.Response(200,json={'status':'completed','rounds':2,'tool_calls':2 if extra_failure else 1,'answer':'Supported [W1]',
                'citations':[],'web_citations':[],'insufficient_evidence':False,'trace_id':root.trace_id})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler),base_url='http://test') as client:
            session=SimpleNamespace(client=client,recorder=recorder,capture=capture,search_cache=cache,
                                    manifest={'verification_tier':'controlled'})
            opened.append(session)
            yield session
    return factory,opened


def test_comparison_keeps_priming_separate_and_compares_actual_queries(tmp_path):
    from app.evaluation.agent_cache_benchmark import compare_agent_cache
    factory,opened=fake_sessions()
    report=asyncio.run(compare_agent_cache(settings=Settings(_env_file=None,deepseek_api_key='test'),
        workload=workload(),repeats=2,output_dir=tmp_path/'run',session_factory=factory,verification_tier='controlled'))
    assert list(report['phases'])==['bypass','cold','priming','warm']
    assert [report['phases'][p]['case_runs'] for p in report['phases']]==[2,2,1,2]
    assert len(opened)==3 and opened[1].search_cache.resets==2
    assert report['comparison']['all_warm_profiles_match_priming'] is True
    assert report['comparison']['all_warm_visible_sources_match_priming'] is True
    assert report['comparison']['paired_descriptive_comparison_eligible'] is True
    assert report['manual_review']['status']=='pending'


def test_query_changes_and_failed_answers_are_not_silently_scored_as_comparable(tmp_path):
    from app.evaluation.agent_cache_benchmark import compare_agent_cache
    factory,_=fake_sessions(changed_query=True,fail=True)
    report=asyncio.run(compare_agent_cache(settings=Settings(_env_file=None,deepseek_api_key='test'),
        workload=workload(),repeats=1,output_dir=tmp_path/'run',session_factory=factory,verification_tier='controlled'))
    assert report['comparison']['all_warm_profiles_match_priming'] is False
    assert report['phases']['warm']['http_failures']==1
    assert report['comparison']['paired_descriptive_comparison_eligible'] is False
    assert (tmp_path/'run/warm/cases.jsonl').exists()


def test_comparison_rejects_existing_output_before_opening_services(tmp_path):
    from app.evaluation.agent_cache_benchmark import compare_agent_cache
    @asynccontextmanager
    async def never(**kwargs):pytest.fail('No services should be opened');yield
    with pytest.raises(FileExistsError):
        asyncio.run(compare_agent_cache(settings=Settings(_env_file=None,deepseek_api_key='test'),workload=workload(),
            repeats=1,output_dir=tmp_path,session_factory=never,verification_tier='controlled'))


def test_same_query_does_not_hide_changed_source_snapshots(tmp_path):
    from app.evaluation.agent_cache_benchmark import compare_agent_cache
    factory,_=fake_sessions(changed_source=True)
    report=asyncio.run(compare_agent_cache(settings=Settings(_env_file=None,deepseek_api_key='test'),
        workload=workload(),repeats=1,output_dir=tmp_path/'run',session_factory=factory,verification_tier='controlled'))
    assert report['comparison']['all_warm_profiles_match_priming'] is True
    assert report['comparison']['all_warm_visible_sources_match_priming'] is False
    assert report['comparison']['paired_descriptive_comparison_eligible'] is False


def test_same_query_and_warm_snapshot_do_not_hide_source_changes_between_phases(tmp_path):
    from app.evaluation.agent_cache_benchmark import compare_agent_cache
    factory,_=fake_sessions(phase_source_change=True)
    report=asyncio.run(compare_agent_cache(settings=Settings(_env_file=None,deepseek_api_key='test'),
        workload=workload(),repeats=1,output_dir=tmp_path/'run',session_factory=factory,verification_tier='controlled'))
    assert report['comparison']['all_warm_profiles_match_priming'] is True
    assert report['comparison']['all_warm_visible_sources_match_priming'] is True
    assert report['comparison']['paired_descriptive_comparison_eligible'] is False


def test_failed_search_queries_are_included_in_actual_request_comparison(tmp_path):
    from app.evaluation.agent_cache_benchmark import compare_agent_cache
    factory,_=fake_sessions(failed_search_first=True)
    report=asyncio.run(compare_agent_cache(settings=Settings(_env_file=None,deepseek_api_key='test'),
        workload=workload(),repeats=1,output_dir=tmp_path/'run',session_factory=factory,verification_tier='controlled'))
    assert report['comparison']['actual_profiles_match_bypass'] is False
    assert report['comparison']['paired_descriptive_comparison_eligible'] is False
