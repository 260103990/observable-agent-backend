import json

from fastapi.testclient import TestClient
import httpx
import pytest

from app.config import Settings
from app.main import app
from tests.test_agent_loop_api import environment, message, call


def search_response():
    return {'results':[{'title':'Provider docs','url':'https://example.org/docs','snippet':'Search snippet.',
        'summary':'Tool calling uses a function name and validated arguments.','content':'',
        'date':None,'rerank_score':2.0,'authority':3}]}


def final(ids=('W1',)):
    return message(content=json.dumps({'claims':[{'text':'工具调用包含函数名与参数。','evidence_ids':list(ids)}],
                                       'insufficient_evidence':False,'reason':None}))


def test_web_request_and_server_flags_are_both_required(environment):
    e=environment
    response=TestClient(app).post('/agent/loop',json={'prompt':'Find docs','enable_web_search':True})
    assert response.status_code==503 and e.requests==[]
    e.settings.web_search_enabled=True
    e.answers.append(message(content='Hello'))
    response=TestClient(app).post('/agent/loop',json={'prompt':'Hello'})
    assert response.status_code==200
    assert {t['function']['name'] for t in e.requests[0]['tools']}=={'add_numbers'}


def test_real_loop_with_controlled_search_and_backend_web_citations(environment):
    e=environment;e.settings.web_search_enabled=True
    e.answers.extend([message(calls=[call('c1','web_search',{'query':'function calling site:example.org','count':1})]),
                      search_response(),final()])
    response=TestClient(app).post('/agent/loop',json={'prompt':'Find official tool calling docs','enable_web_search':True},
                                  headers={'X-Request-ID':'web-request'})
    assert response.status_code==200,response.text
    body=response.json()
    assert body['rounds']==2 and body['tool_calls']==1 and body['answer'].endswith('[W1]')
    assert body['citations']==[] and body['web_citations'][0]['url']=='https://example.org/docs'
    assert body['web_citations'][0]['source_type']=='web_search_result'
    assert e.requests[1]['need_content'] is False and e.requests[1]['localization']==''
    search_trace=TestClient(app).get('/traces/'+body['web_citations'][0]['search_trace_id']).json()
    assert search_trace['spans'][0]['name']=='web.search' and search_trace['request_id']=='web-request'
    assert search_trace['spans'][0]['attributes']['outbound_request_attempts']==1
    assert e.clients[0].is_closed


def test_paper_and_web_tools_available_together_and_citations_remain_separate(environment):
    e=environment;e.settings.web_search_enabled=True
    e.answers.extend([message(calls=[call('p1','paper_retrieval',{'query':'baseline'}),
                                    call('w1','web_search',{'query':'baseline docs','count':1})]),search_response(),final(('E1','W1'))])
    response=TestClient(app).post('/agent/loop',json={'prompt':'Compare with external sources','paper_id':'paper-1','enable_web_search':True})
    assert response.status_code==200,response.text
    body=response.json()
    assert body['citations'][0]['page_number']==10 and body['web_citations'][0]['evidence_id']=='W1'
    assert {t['function']['name'] for t in e.requests[0]['tools']}=={'add_numbers','paper_retrieval','web_search'}


def test_failed_search_can_be_reported_as_insufficient_without_fake_links(environment):
    e=environment;e.settings.web_search_enabled=True
    e.answers.extend([message(calls=[call('w1','web_search',{'query':'q','count':1})]),httpx.ReadTimeout('private'),
        message(content=json.dumps({'claims':[],'insufficient_evidence':True,'reason':'Search timed out; no support available.'}))])
    response=TestClient(app).post('/agent/loop',json={'prompt':'q','enable_web_search':True})
    assert response.status_code==200,response.text
    assert response.json()['insufficient_evidence'] and response.json()['web_citations']==[]
    assert 'private' not in response.text
    assert len(e.requests)==3  # two model turns and one Search attempt, no retry


def test_web_evidence_does_not_leak_between_requests(environment):
    e=environment;e.settings.web_search_enabled=True
    e.answers.extend([message(calls=[call('w1','web_search',{'query':'q','count':1})]),search_response(),final(),final(),final()])
    client=TestClient(app)
    assert client.post('/agent/loop',json={'prompt':'q','enable_web_search':True}).status_code==200
    assert client.post('/agent/loop',json={'prompt':'q','enable_web_search':True}).status_code==502


@pytest.mark.parametrize('flag',['true',1,None])
def test_web_flag_is_strict_boolean(environment,flag):
    response=TestClient(app).post('/agent/loop',json={'prompt':'q','enable_web_search':flag})
    assert response.status_code==422 and environment.requests==[]


def test_web_defaults_and_budget_configuration():
    s=Settings(_env_file=None)
    assert s.web_search_enabled is False and s.web_search_timeout_seconds==30
    assert s.web_search_max_response_bytes==262144
    for values in ({'web_search_timeout_seconds':0},{'web_search_timeout_seconds':float('inf')},
                   {'web_search_max_evidence_items':0},{'web_search_max_response_bytes':-1}):
        with pytest.raises(ValueError): Settings(_env_file=None,**values)
