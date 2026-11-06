import asyncio
import json

import httpx
import pytest

from app.observability.models import SpanRecord, TraceRecord


def workload_dict():
    return {'version':'v2.3-test','paper_sha256':'a'*64,'cases':[{
        'id':'calc','prompt':'Compute 17+25','required_tools':['add_numbers'],'allowed_tools':['add_numbers'],
        'answer_terms':[['42']]}]}


def trace(identifier='root',name='agent.loop',**attributes):
    return TraceRecord(trace_id=identifier,request_id='request',status='ok',duration_ms=50,
        spans=[SpanRecord(span_id='root',parent_span_id=None,name=name,status='ok',duration_ms=50,
                          attributes=attributes,error_type=None)])


def test_workload_validates_scope_duplicates_and_expected_terms():
    from app.evaluation.models import Workload
    data=workload_dict()
    assert Workload.model_validate(data).cases[0].to_request(None)=={'prompt':'Compute 17+25','enable_web_search':False,'top_k':5}
    bad=[{**data,'cases':data['cases']*2},
         {**data,'cases':[{**data['cases'][0],'required_tools':['paper_retrieval']}]},
         {**data,'cases':[{**data['cases'][0],'required_tools':['web_search'],'enable_web_search':False}]},
         {**data,'cases':[{**data['cases'][0],'answer_terms':[[]]}]},
         {**data,'unexpected':'field'}]
    for value in bad:
        with pytest.raises(ValueError): Workload.model_validate(value)


def test_paper_case_requires_explicit_selected_paper():
    from app.evaluation.models import EvaluationCase
    case=EvaluationCase(id='paper',prompt='Models?',use_paper=True,required_tools=['paper_retrieval'])
    with pytest.raises(ValueError):case.to_request(None)
    assert case.to_request('selected')['paper_id']=='selected'


def test_recorded_traces_are_independent_and_filtered_by_request_id():
    from app.evaluation.recording import EvaluationRecorder
    async def run():
        recorder=EvaluationRecorder()
        original=trace()
        await recorder.save(original)
        original.spans[0].attributes['changed']=True
        rows=await recorder.for_request('request')
        assert 'changed' not in rows[0].spans[0].attributes
        rows[0].spans.clear()
        assert len((await recorder.get('root')).spans)==1
        assert await recorder.for_request('other')==[]
    asyncio.run(run())


def test_request_hooks_capture_all_search_parameters_and_tool_evidence_without_headers():
    from app.evaluation.recording import EvaluationCapture
    capture=EvaluationCapture(model_url='https://api.deepseek.com/chat/completions')
    async def run():
        with capture.for_request('request'):
            request=httpx.Request('POST','https://api.deepseek.com/v1/tools/search',
                headers={'Authorization':'Bearer private-secret'},json={'query':'cache','count':2,'need_content':False,'localization':''})
            await capture.on_request(request)
            await capture.on_response(httpx.Response(200,request=request))
            assistant={'role':'assistant','content':None,'tool_calls':[{'id':'c1','type':'function',
                       'function':{'name':'paper_retrieval','arguments':'{"query":"models"}'}}]}
            tool={'role':'tool','tool_call_id':'c1','content':json.dumps({'ok':True,'output':{
                  'evidence':[{'evidence_id':'E1','page_number':10,'text':'Search-R1 model.'}]},'attempts':1})}
            for _ in range(2):
                await capture.on_request(httpx.Request('POST','https://api.deepseek.com/chat/completions',
                    headers={'Authorization':'Bearer private-secret'},json={'messages':[assistant,tool]}))
        rows=capture.for_case('request')
        assert len(rows['search_requests'])==1 and rows['search_requests'][0]['parameters']['count']==2
        assert rows['search_requests'][0]['http_status']==200
        assert len(rows['tool_results'])==1 and rows['tool_results'][0]['arguments']=={'query':'models'}
        assert rows['tool_results'][0]['result']['output']['evidence'][0]['page_number']==10
        assert 'Authorization' not in json.dumps(rows) and 'private-secret' not in json.dumps(rows)
        assert capture.for_case('other')['search_requests']==[]
    asyncio.run(run())


def test_metric_aggregation_counts_uncited_searches_once_and_separates_nested_durations():
    from app.evaluation.scoring import summarize_traces
    root=trace(rounds=2,tool_calls=2)
    root.spans.extend([
        SpanRecord(span_id='m',parent_span_id='root',name='model.turn',status='ok',duration_ms=10,attributes={},error_type=None),
        SpanRecord(span_id='t',parent_span_id='root',name='tool.execute',status='ok',duration_ms=20,
                   attributes={'tool_name':'web_search','attempts':1},error_type=None)])
    one=trace('search-1','web.search',outbound_request_attempts=1,response_body_bytes=100,result_count=1)
    two=trace('search-2','web.search',outbound_request_attempts=1,response_body_bytes=200,result_count=0)
    for item in (one,two):
        item.spans.append(SpanRecord(span_id='s',parent_span_id='root',name='search.http',status='ok',duration_ms=5,
                         attributes={'response_body_bytes':999},error_type=None))
    metrics=summarize_traces([root,one,two],root_trace_id='root')
    assert metrics['search_request_attempts']==2 and metrics['search_response_body_bytes']==300
    assert metrics['model_duration_ms']==10 and metrics['tool_duration_ms']==20
    assert metrics['search_http_duration_ms']==10 and metrics['root_duration_ms']==50


def test_model_visible_support_pages_and_final_cited_pages_are_distinct():
    from app.evaluation.models import EvaluationCase
    from app.evaluation.scoring import score_case
    case=EvaluationCase(id='paper',prompt='Datasets?',use_paper=True,required_tools=['paper_retrieval'],
                        expected_pages=[13],answer_terms=[['HotpotQA']])
    result={'status':'completed','answer':'HotpotQA [E1]','tool_calls':1,'citations':[{'page_number':10}],
            'web_citations':[],'insufficient_evidence':False}
    observations={'tool_results':[{'call_id':'c1','name':'paper_retrieval','result':{'ok':True,'output':{
                 'evidence':[{'page_number':10},{'page_number':13}]}}}],'search_requests':[]}
    checked=score_case(case,result,{'tool_calls_by_name':{'paper_retrieval':1}},observations)
    assert checked['checks']['model_visible_support_pages_present'] is True
    assert checked['checks']['cited_support_pages_present'] is False
    assert checked['manual_review']['status']=='pending'
    assert checked['checks']['answer_terms_present'] is True


def test_source_domain_checks_use_hostname_boundary_not_substring():
    from app.evaluation.models import EvaluationCase
    from app.evaluation.scoring import score_case
    case=EvaluationCase(id='web',prompt='Docs?',enable_web_search=True,required_tools=['web_search'],
                        expected_web_domains=['api-docs.deepseek.com'])
    base={'status':'completed','answer':'Result','tool_calls':1,'citations':[],
          'web_citations':[{'url':'https://api-docs.deepseek.com.evil.example/a'}]}
    checked=score_case(case,base,{'tool_calls_by_name':{'web_search':1}},{'tool_results':[],'search_requests':[]})
    assert checked['checks']['expected_web_domain_present'] is False


def test_unobserved_last_round_tool_is_marked_incomplete_not_fabricated():
    from app.evaluation.models import EvaluationCase
    from app.evaluation.scoring import score_case
    result={'status':'limit_reached','tool_calls':1,'answer':None,'citations':[],'web_citations':[]}
    checked=score_case(EvaluationCase(id='limited',prompt='q'),result,{'tool_calls_by_name':{'add_numbers':1}},
                       {'tool_results':[],'search_requests':[]})
    assert checked['tool_observation_complete'] is False
    assert checked['manual_review']['status']=='pending'


def test_evaluation_continues_after_http_failure_and_writes_machine_and_readable_reports(tmp_path):
    from app.evaluation.models import Workload
    from app.evaluation.recording import EvaluationCapture,EvaluationRecorder
    from app.evaluation.runner import evaluate_cases
    data=workload_dict();data['cases'].append({**data['cases'][0],'id':'second'})
    recorder=EvaluationRecorder();capture=EvaluationCapture(model_url='https://api.deepseek.com/chat/completions')
    requests=[]
    async def handler(request):
        requests.append(request)
        if len(requests)==1:return httpx.Response(503,json={'detail':'Unavailable'})
        return httpx.Response(200,json={'status':'completed','answer':'42','rounds':1,'tool_calls':0,
                                       'citations':[],'web_citations':[],'trace_id':'missing'})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler),base_url='http://eval.local') as client:
            return await evaluate_cases(client=client,workload=Workload.model_validate(data),recorder=recorder,
                capture=capture,paper_id=None,output_dir=tmp_path,manifest={'verification_tier':'controlled'})
    report=asyncio.run(run())
    assert len(report['results'])==2 and report['results'][0]['http_status']==503
    assert report['results'][1]['http_status']==200
    assert len({r.headers['x-request-id'] for r in requests})==2
    assert (tmp_path/'report.json').is_file() and (tmp_path/'report.md').is_file()
    assert len((tmp_path/'cases.jsonl').read_text(encoding='utf-8').splitlines())==2
    assert report['manual_review_pending']==2 and report['automated_pass_count']==0


def test_report_directory_must_be_new_to_avoid_overwriting_previous_runs(tmp_path):
    from app.evaluation.models import Workload
    from app.evaluation.recording import EvaluationCapture,EvaluationRecorder
    from app.evaluation.runner import evaluate_cases
    (tmp_path/'report.json').write_text('existing',encoding='utf-8')
    async def run():
        return await evaluate_cases(client=None,workload=Workload.model_validate(workload_dict()),
            recorder=EvaluationRecorder(),capture=EvaluationCapture(model_url='https://model.invalid'),
            paper_id=None,output_dir=tmp_path,manifest={})
    with pytest.raises(FileExistsError):asyncio.run(run())
    assert (tmp_path/'report.json').read_text(encoding='utf-8')=='existing'


def test_live_runtime_requires_explicit_readonly_paper_connection_before_resources(monkeypatch):
    from app.config import Settings
    from app.evaluation.models import Workload
    from app.evaluation.runtime import evaluation_session
    data=workload_dict();data['cases'][0].update(use_paper=True,required_tools=['paper_retrieval'],allowed_tools=['paper_retrieval'])
    async def run():
        with pytest.raises(ValueError):
            async with evaluation_session(settings=Settings(_env_file=None,deepseek_api_key='fake'),
                                          workload=Workload.model_validate(data),paper_id='paper',database_url=None):
                pytest.fail('An implicit application DB must never be selected')
    asyncio.run(run())


def test_runtime_uses_readonly_pool_and_restores_application_resources_on_error(monkeypatch):
    from app.config import Settings
    from app.evaluation.models import Workload
    from app.evaluation import runtime
    from app.main import app
    from app.rag.storage_models import PaperInfo,EmbeddingSpec
    data=workload_dict();data['cases'][0].update(use_paper=True,required_tools=['paper_retrieval'],allowed_tools=['paper_retrieval'])
    info=PaperInfo(paper_id='paper',title='Cortex',source_path='unused.pdf',source_sha256='a'*64,
                   embedding=EmbeddingSpec(model='embed',revision='r',query_instruction='find'))
    events=[]
    class Pool:
        async def close(self):events.append('closed')
    async def pool_factory(**kwargs):
        assert kwargs['server_settings']['default_transaction_read_only']=='on'
        events.append('opened');return Pool()
    class Store:
        def __init__(self,pool):pass
        async def get_info(self,paper_id):return info
    monkeypatch.setattr(runtime.asyncpg,'create_pool',pool_factory)
    monkeypatch.setattr(runtime,'PostgresPaperStore',Store)
    previous=dict(app.dependency_overrides)
    async def run():
        with pytest.raises(RuntimeError,match='case interrupted'):
            async with runtime.evaluation_session(settings=Settings(_env_file=None,deepseek_api_key='fake'),
                workload=Workload.model_validate(data),paper_id='paper',database_url='postgresql://test/readonly') as session:
                assert session.paper_info.source_sha256=='a'*64 and hasattr(app.state,'rag_service')
                assert session.manifest['database_mode']=='read_only_existing_paper'
                raise RuntimeError('case interrupted')
        assert app.dependency_overrides==previous and not hasattr(app.state,'rag_service')
        assert events==['opened','closed']
    asyncio.run(run())


def test_wrong_paper_hash_is_rejected_before_provider_execution(monkeypatch):
    from app.config import Settings
    from app.evaluation.models import Workload
    from app.evaluation import runtime
    from app.rag.storage_models import PaperInfo,EmbeddingSpec
    data=workload_dict();data['cases'][0].update(use_paper=True,required_tools=['paper_retrieval'],allowed_tools=['paper_retrieval'])
    class Pool:
        async def close(self):pass
    async def factory(**kwargs):return Pool()
    class Store:
        def __init__(self,pool):pass
        async def get_info(self,paper_id):
            return PaperInfo(paper_id='paper',title='Wrong',source_path='unused',source_sha256='b'*64,
                embedding=EmbeddingSpec(model='embed',revision='r',query_instruction='find'))
    monkeypatch.setattr(runtime.asyncpg,'create_pool',factory)
    monkeypatch.setattr(runtime,'PostgresPaperStore',Store)
    async def run():
        with pytest.raises(ValueError):
            async with runtime.evaluation_session(settings=Settings(_env_file=None,deepseek_api_key='fake'),
                workload=Workload.model_validate(data),paper_id='paper',database_url='postgresql://test/readonly'):
                pytest.fail('Wrong source paper must not be evaluated')
    asyncio.run(run())


@pytest.mark.parametrize('recovered',[False,True])
def test_negative_case_requires_successful_retrieval_not_merely_service_failure(recovered):
    from app.evaluation.models import EvaluationCase
    from app.evaluation.scoring import score_case
    case=EvaluationCase(id='negative',prompt='Mars experiment?',use_paper=True,
                        required_tools=['paper_retrieval'],expected_insufficient=True)
    rows=[{'call_id':'failed','name':'paper_retrieval','result':{'ok':False,'error':'tool_failed','attempts':1}}]
    if recovered:rows.append({'call_id':'recovered','name':'paper_retrieval','result':{'ok':True,'output':{'evidence':[]},'attempts':1}})
    result={'status':'completed','answer':'No support','tool_calls':len(rows),'citations':[],
            'web_citations':[],'insufficient_evidence':True}
    scored=score_case(case,result,{'root_trace_present':True,'tool_calls_by_name':{'paper_retrieval':len(rows)}},
                      {'tool_results':rows,'search_requests':[]})
    assert scored['checks']['required_tools_succeeded'] is recovered
    assert scored['automated_pass'] is recovered


def test_failed_http_case_keeps_trace_header_and_error_stage_metrics(tmp_path):
    from app.evaluation.models import Workload
    from app.evaluation.recording import EvaluationRecorder,EvaluationCapture
    from app.evaluation.runner import evaluate_cases
    recorder=EvaluationRecorder()
    async def handler(request):
        root=trace('failed-root')
        root.request_id=request.headers['x-request-id'];root.status='error';root.duration_ms=100
        root.spans.append(SpanRecord(span_id='model',parent_span_id='root',name='model.turn',status='error',
            duration_ms=90,attributes={},error_type='LoopTimeoutError'))
        await recorder.save(root)
        return httpx.Response(504,json={'detail':'Timed out'},headers={'X-Trace-ID':'failed-root'})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler),base_url='http://eval.local') as client:
            return await evaluate_cases(client=client,workload=Workload.model_validate(workload_dict()),recorder=recorder,
                capture=EvaluationCapture(model_url='https://api.deepseek.com/chat/completions'),paper_id=None,
                output_dir=tmp_path,manifest={'verification_tier':'controlled'})
    row=asyncio.run(run())['results'][0]
    assert row['http_status']==504 and row['metrics']['root_trace_present'] is True
    assert row['metrics']['root_duration_ms']==100 and row['metrics']['model_duration_ms']==90
    assert not row['automated_pass']


def test_validated_model_decisions_are_kept_for_finalizer_failure_diagnosis():
    from app.agent.loop_models import LoopTurn
    from app.evaluation.recording import EvaluationCapture
    capture=EvaluationCapture(model_url='https://api.deepseek.com/chat/completions')
    with capture.for_request('request'):
        capture.record_decision(LoopTurn(content='{"claims":[],"insufficient_evidence":true,"reason":"text [E1]"}'))
    decisions=capture.for_case('request')['model_decisions']
    assert len(decisions)==1 and 'text [E1]' in decisions[0]['content']


@pytest.mark.parametrize('observed,expected',[(0,False),(1,True)])
def test_failed_http_observation_completeness_uses_recorded_execution_count(observed,expected):
    from app.evaluation.models import EvaluationCase
    from app.evaluation.scoring import score_case
    rows=[{'call_id':'c1','name':'paper_retrieval','result':{'ok':True,'output':{'evidence':[]}}}]*observed
    checked=score_case(EvaluationCase(id='failed',prompt='q'),{},
        {'root_trace_present':True,'logical_tool_calls':1,'tool_calls_by_name':{'paper_retrieval':1}},
        {'tool_results':rows,'capture_errors':[]})
    assert checked['tool_observation_complete'] is expected


def test_no_result_and_no_execution_trace_have_unknown_observation_completeness():
    from app.evaluation.models import EvaluationCase
    from app.evaluation.scoring import score_case
    scored=score_case(EvaluationCase(id='missing',prompt='q'),{},
        {'root_trace_present':False,'logical_tool_calls':None,'tool_calls_by_name':{}},
        {'tool_results':[],'capture_errors':[]})
    assert scored['tool_observation_complete'] is None and not scored['automated_pass']
