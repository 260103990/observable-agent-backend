import json

from fastapi.testclient import TestClient
import httpx

from app.main import app
from tests.test_agent_loop_api import environment,message,call


BAD=json.dumps({'claims':[],'insufficient_evidence':False,'reason':'No support.'})
GOOD=json.dumps({'claims':[],'insufficient_evidence':True,'reason':'No support.'})


def test_http_repair_uses_same_client_and_reports_final_repair_count(environment):
    e=environment;e.answers.extend([message(content=BAD),message(content=GOOD)])
    response=TestClient(app).post('/agent/loop',json={'prompt':'Unsupported question','paper_id':'paper-1'})
    assert response.status_code==200,response.text
    body=response.json()
    assert body['final_repairs']==1 and body['rounds']==2 and body['tool_calls']==0
    assert body['insufficient_evidence'] and body['citations']==[]
    assert json.loads(e.requests[1]['messages'][-2]['content'])['insufficient_evidence'] is False
    assert len(e.clients)==1 and e.clients[0].is_closed
    trace=TestClient(app).get('/traces/'+body['trace_id']).json()
    assert trace['spans'][0]['attributes']['final_repairs']==1
    assert [s['status'] for s in trace['spans'] if s['name']=='answer.validate']==['error','ok']


def test_http_repair_can_be_disabled_and_invalid_output_still_502(environment):
    e=environment;e.settings.agent_loop_max_final_repairs=0
    e.answers.append(message(content=BAD))
    response=TestClient(app).post('/agent/loop',json={'prompt':'q','paper_id':'paper-1'})
    assert response.status_code==502 and len(e.requests)==1 and e.clients[0].is_closed


def test_http_repair_timeout_still_returns_504_and_closes_client(environment):
    e=environment;e.answers.extend([message(content=BAD),httpx.ReadTimeout('private provider data')])
    response=TestClient(app).post('/agent/loop',json={'prompt':'q','paper_id':'paper-1'})
    assert response.status_code==504 and 'private' not in response.text
    assert len(e.requests)==2 and e.clients[0].is_closed


def test_http_repair_tool_call_is_rejected_before_execution(environment):
    e=environment;e.answers.extend([message(content=BAD),message(calls=[call('new','add_numbers',{'a':1,'b':2})])])
    response=TestClient(app).post('/agent/loop',json={'prompt':'q','paper_id':'paper-1'})
    assert response.status_code==502
    trace=TestClient(app).get('/traces/'+response.headers['x-trace-id']).json()
    assert not [span for span in trace['spans'] if span['name']=='tool.execute']
