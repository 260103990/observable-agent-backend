import asyncio
import json

from fastapi.testclient import TestClient
import httpx
import pytest

from app.main import app
from tests.answer_helpers import hit
from tests.test_agent_loop_api import environment,message,call
from tests.test_deepseek_answer import make_answer,envelope
from tests.test_web_loop_api import search_response,final


MARKER='Answer the requested facts and scope'


def test_direct_paper_answer_sends_focus_policy_with_original_evidence_contract():
    seen=[]
    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200,json=envelope())
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            result=await make_answer(http,focus_policy_enabled=True).answer('paper-1','Throughput?',[hit()])
            assert result.citations[0].raw_text==hit().chunk.raw_text and not http.is_closed
    asyncio.run(scenario())
    assert MARKER in seen[0]['messages'][0]['content']
    assert json.loads(seen[0]['messages'][1]['content'])['evidence'][0]['text']==hit().chunk.text
    assert seen[0]['response_format']=={'type':'json_object'}


@pytest.mark.parametrize('mode',['paper','web','mixed'])
def test_grounded_loop_modes_receive_same_focus_policy_and_preserve_sources(environment,mode):
    e=environment
    e.settings.answer_scope_policy_enabled=True
    body={'prompt':'Requested fact?'}
    if mode!='web':body['paper_id']='paper-1'
    if mode=='paper':
        e.answers.extend([message(calls=[call('p1','paper_retrieval',{'query':'fact'})]),
            message(content=json.dumps({'claims':[{'text':'A supported fact.','evidence_ids':['E1']}],
                                        'insufficient_evidence':False,'reason':None}))])
    else:
        e.settings.web_search_enabled=True;body['enable_web_search']=True
        e.answers.extend([message(calls=[call('w1','web_search',{'query':'fact','count':1})]),search_response(),final()])
    response=TestClient(app).post('/agent/loop',json=body)
    assert response.status_code==200,response.text
    assert MARKER in e.requests[0]['messages'][0]['content']
    assert response.json()['rounds']==2 and response.json()['tool_calls']==1
    assert all(client.is_closed for client in e.clients)


def test_final_repair_keeps_focus_policy_without_repeating_tools(environment):
    e=environment
    e.settings.answer_scope_policy_enabled=True
    e.answers.extend([message(content=json.dumps({'claims':[],'insufficient_evidence':False,'reason':'No support'})),
                      message(content=json.dumps({'claims':[],'insufficient_evidence':True,'reason':'Missing requested support'}))])
    response=TestClient(app).post('/agent/loop',json={'prompt':'Requested fact?','paper_id':'paper-1'})
    assert response.status_code==200,response.text
    assert response.json()['tool_calls']==0 and response.json()['final_repairs']==1
    assert MARKER in e.requests[1]['messages'][-1]['content']
    assert len(e.clients)==1 and e.clients[0].is_closed


def test_plain_calculator_loop_does_not_receive_grounded_answer_policy(environment):
    e=environment
    e.settings.answer_scope_policy_enabled=True
    e.answers.append(message(content='42'))
    assert TestClient(app).post('/agent/loop',json={'prompt':'Hello'}).status_code==200
    assert MARKER not in e.requests[0]['messages'][0]['content']


@pytest.mark.parametrize('mode',['paper','web'])
def test_grounded_policy_is_off_by_default(environment,mode):
    from app.config import Settings
    assert Settings(_env_file=None).answer_scope_policy_enabled is False
    e=environment
    body={'prompt':'A question'}
    if mode=='paper':body['paper_id']='paper-1'
    else:body['enable_web_search']=True;e.settings.web_search_enabled=True
    e.answers.append(message(content=json.dumps({'claims':[],'insufficient_evidence':True,'reason':'Missing support'})))
    response=TestClient(app).post('/agent/loop',json=body)
    assert response.status_code==200,response.text
    assert MARKER not in e.requests[0]['messages'][0]['content']


def test_direct_answer_default_keeps_original_prompt():
    seen=[]
    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200,json=envelope())
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            await make_answer(http).answer('paper-1','Throughput?',[hit()])
    asyncio.run(scenario())
    assert MARKER not in seen[0]['messages'][0]['content']


@pytest.mark.parametrize('enabled',[False,True])
def test_direct_factory_respects_explicit_server_setting(enabled):
    from app.config import Settings
    from app.rag.answer import DeepSeekAnswerClient
    settings=Settings(_env_file=None,deepseek_api_key='unit-test-key',answer_scope_policy_enabled=enabled)
    seen=[]
    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200,json=envelope())
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            await DeepSeekAnswerClient.from_settings(settings,client=http).answer('paper-1','Throughput?',[hit()])
    asyncio.run(scenario())
    assert (MARKER in seen[0]['messages'][0]['content']) is enabled


def test_default_final_repair_also_keeps_original_prompt(environment):
    e=environment
    e.answers.extend([message(content=json.dumps({'claims':[],'insufficient_evidence':False,'reason':'No support'})),
                      message(content=json.dumps({'claims':[],'insufficient_evidence':True,'reason':'Missing support'}))])
    response=TestClient(app).post('/agent/loop',json={'prompt':'A question','paper_id':'paper-1'})
    assert response.status_code==200 and response.json()['final_repairs']==1
    assert MARKER not in e.requests[1]['messages'][-1]['content']
