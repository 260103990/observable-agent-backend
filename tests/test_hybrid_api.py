import json

from fastapi.testclient import TestClient
import pytest

from app.main import app
from tests.test_agent_loop_api import environment,message,call


@pytest.mark.parametrize('mode',['vector','hybrid'])
def test_context_followup_prompt_is_only_enabled_for_selected_hybrid_mode(environment,mode):
    e=environment
    e.settings.rag_retrieval_strategy=mode
    e.answers.extend([message(calls=[call('p1','paper_retrieval',{'query':'datasets'})]),
        message(content=json.dumps({'claims':[{'text':'A supported result.','evidence_ids':['E1']}],
                                    'insufficient_evidence':False,'reason':None}))])
    response=TestClient(app).post('/agent/loop',json={'prompt':'datasets?','paper_id':'paper-1'})
    assert response.status_code==200,response.text
    prompt=e.requests[0]['messages'][0]['content']
    assert ('If a passage refers to a figure or table' in prompt)==(mode=='hybrid')
    assert response.json()['citations'][0]['page_number']==10
    assert e.retrievals[0][2]['top_k']==5
    assert all(client.is_closed for client in e.clients)
