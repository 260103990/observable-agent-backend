import json
from types import SimpleNamespace

from fastapi.testclient import TestClient
import httpx
import pytest

from app.config import Settings, get_settings
from app.main import app
from app.api.dependencies import get_trace_store
from app.observability.store import InMemoryTraceStore
from tests.answer_helpers import hit


@pytest.fixture
def environment(monkeypatch):
    from app.api import loop_dependencies as deps
    clients, requests, retrievals = [], [], []
    traces=InMemoryTraceStore()
    answers=[]
    def transport(request):
        requests.append(json.loads(request.content))
        result=answers.pop(0)
        if isinstance(result,Exception): raise result
        return httpx.Response(200,json=result)
    original=httpx.AsyncClient
    def http_factory(**kwargs):
        client=original(transport=httpx.MockTransport(transport),**kwargs)
        clients.append(client)
        return client
    monkeypatch.setattr(deps,'httpx',SimpleNamespace(AsyncClient=http_factory,Limits=httpx.Limits))
    settings=Settings(_env_file=None,deepseek_api_key='fake-unit-key')
    app.dependency_overrides[get_settings]=lambda:settings
    app.dependency_overrides[get_trace_store]=lambda:traces
    class Rag:
        async def retrieve(self,paper_id,query,**kwargs):
            retrievals.append((paper_id,query,kwargs))
            return [hit(paper_id=paper_id)]
    app.state.rag_service=Rag()
    yield SimpleNamespace(settings=settings,clients=clients,requests=requests,answers=answers,
                          traces=traces,retrievals=retrievals)
    app.dependency_overrides.clear()
    if hasattr(app.state,'rag_service'): del app.state.rag_service


def message(*,content=None,calls=None):
    return {'id':'model-1','model':'deepseek-flash','choices':[{'finish_reason':'tool_calls' if calls else 'stop',
            'message':{'role':'assistant','content':content,'tool_calls':calls}}]}


def call(identifier,name,arguments):
    return {'id':identifier,'type':'function','function':{'name':name,'arguments':json.dumps(arguments)}}


def test_http_real_loop_model_adapter_tool_and_trace_work_together(environment):
    e=environment
    e.answers.extend([message(calls=[call('c1','add_numbers',{'a':17,'b':25})]),
                      message(calls=[call('c2','add_numbers',{'a':42,'b':42})]),message(content='84')])
    response=TestClient(app).post('/agent/loop',json={'prompt':'Calculate (17 + 25) + 42'},headers={'X-Request-ID':'api-loop'})
    assert response.status_code==200,response.text
    body=response.json()
    assert body['answer']=='84' and body['rounds']==3 and body['tool_calls']==2
    assert response.headers['x-trace-id']==body['trace_id'] and response.headers['x-request-id']=='api-loop'
    trace=TestClient(app).get('/traces/'+body['trace_id']).json()
    assert trace['status']=='ok' and trace['request_id']=='api-loop'
    assert len(e.clients)==1 and e.clients[0].is_closed
    assert json.loads(e.requests[1]['messages'][-1]['content'])['output']==42


def test_http_paper_scope_returns_backend_citations(environment):
    e=environment
    e.answers.extend([message(calls=[call('c1','paper_retrieval',{'query':'baseline throughput'})]),
        message(content=json.dumps({'claims':[{'text':'吞吐量为 42 requests/s。','evidence_ids':['E1']}],
                                    'insufficient_evidence':False,'reason':None}))])
    response=TestClient(app).post('/agent/loop',json={'prompt':'基线吞吐量是什么？','paper_id':'paper-1','top_k':3})
    assert response.status_code==200,response.text
    body=response.json()
    assert body['citations'][0]['paper_id']=='paper-1' and body['citations'][0]['page_number']==10
    assert body['citations'][0]['raw_text']==hit().chunk.raw_text
    assert e.retrievals[0][0]=='paper-1' and e.retrievals[0][2]['top_k']==3
    assert {t['function']['name'] for t in e.requests[0]['tools']}=={'add_numbers','paper_retrieval'}


def test_each_http_request_has_independent_evidence_ledger(environment):
    e=environment
    claim=json.dumps({'claims':[{'text':'A claim','evidence_ids':['E1']}],'insufficient_evidence':False,'reason':None})
    e.answers.extend([message(calls=[call('c1','paper_retrieval',{'query':'q'})]),message(content=claim),message(content=claim),message(content=claim)])
    client=TestClient(app)
    assert client.post('/agent/loop',json={'prompt':'q','paper_id':'paper-1'}).status_code==200
    assert client.post('/agent/loop',json={'prompt':'q','paper_id':'paper-1'}).status_code==502
    assert all(c.is_closed for c in e.clients)


def test_http_limits_return_explicit_outcome_without_extra_model_call(environment):
    e=environment
    e.settings.agent_loop_max_rounds=1
    e.answers.append(message(calls=[call('c1','add_numbers',{'a':1,'b':2})]))
    response=TestClient(app).post('/agent/loop',json={'prompt':'Compute'})
    assert response.status_code==200,response.text
    assert response.json()['status']=='limit_reached' and response.json()['stop_reason']=='max_rounds'
    assert response.json()['answer'] is None and len(e.requests)==1


@pytest.mark.parametrize('body',[
    {'prompt':' '},{'prompt':'x'*4001},{'prompt':'q','paper_id':' '},
    {'prompt':'q','top_k':0},{'prompt':'q','top_k':True},{'prompt':'q','top_k':101},
    {'prompt':'q','unexpected':'field'},
])
def test_http_invalid_input_is_rejected_before_model(environment,body):
    response=TestClient(app).post('/agent/loop',json=body)
    assert response.status_code==422 and environment.requests==[]


@pytest.mark.parametrize('missing',['key','rag'])
def test_missing_configuration_is_safe_503(environment,missing):
    e=environment
    if missing=='key': e.settings.deepseek_api_key=None
    else: del app.state.rag_service
    response=TestClient(app).post('/agent/loop',json={'prompt':'q','paper_id':'paper-1'} if missing=='rag' else {'prompt':'q'})
    assert response.status_code==503,response.text
    assert e.requests==[]


@pytest.mark.parametrize('failure,status',[(httpx.ReadTimeout('private'),504),(httpx.ConnectError('private'),503),({},502)])
def test_http_provider_errors_are_mapped_safely_and_client_closes(environment,failure,status):
    environment.answers.append(failure)
    response=TestClient(app).post('/agent/loop',json={'prompt':'q'})
    assert response.status_code==status and 'private' not in response.text
    assert environment.clients[0].is_closed and 'x-trace-id' in response.headers


def test_new_settings_have_finite_bounded_defaults():
    s=Settings(_env_file=None)
    assert s.agent_loop_max_rounds==5 and s.agent_loop_max_tool_calls==8
    for kwargs in ({'agent_loop_max_rounds':0},{'agent_loop_max_rounds':13},
                   {'agent_loop_run_timeout_seconds':float('inf')},{'agent_loop_tool_timeout_seconds':0}):
        with pytest.raises(ValueError): Settings(_env_file=None,**kwargs)
