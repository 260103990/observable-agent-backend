import json

from fastapi.testclient import TestClient

from app.main import app
from tests.test_agent_loop_api import environment, message, call
from tests.test_web_loop_api import search_response, final


def tool_outputs(request):
    return [json.loads(m['content']) for m in request['messages'] if m['role'] == 'tool']


def test_paper_history_reuses_exact_evidence_and_preserves_citation(environment):
    e = environment
    e.answers.extend([
        message(calls=[call('p1', 'paper_retrieval', {'query': 'first'})]),
        message(calls=[call('p2', 'paper_retrieval', {'query': 'second'})]),
        message(content=json.dumps({'claims': [{'text': 'Throughput is 42 requests/s.', 'evidence_ids': ['E1']}],
                                    'insufficient_evidence': False, 'reason': None})),
    ])
    response = TestClient(app).post('/agent/loop', json={'prompt': 'Throughput?', 'paper_id': 'paper-1'})
    assert response.status_code == 200, response.text
    outputs = tool_outputs(e.requests[-1])
    assert outputs[0]['output']['evidence'] == [{'evidence_id': 'E1', 'same_as_tool_call_id': 'p2'}]
    assert outputs[1]['output']['evidence'][0]['text']
    assert outputs[0]['output']['retrieval_trace_id'] != outputs[1]['output']['retrieval_trace_id']
    assert response.json()['citations'][0]['page_number'] == 10
    assert len(e.retrievals) == 2 and all(c.is_closed for c in e.clients)


def test_web_history_reuses_snapshot_without_losing_source_metadata(environment):
    e = environment
    e.settings.web_search_enabled = True
    e.answers.extend([
        message(calls=[call('w1', 'web_search', {'query': 'docs', 'count': 1})]), search_response(),
        message(calls=[call('w2', 'web_search', {'query': 'docs', 'count': 1})]), search_response(), final(),
    ])
    response = TestClient(app).post('/agent/loop', json={'prompt': 'Find docs', 'enable_web_search': True})
    assert response.status_code == 200, response.text
    outputs = tool_outputs(e.requests[-1])
    assert outputs[0]['output']['evidence'] == [{'evidence_id': 'W1', 'same_as_tool_call_id': 'w2'}]
    source = outputs[1]['output']['evidence'][0]
    assert source['summary'] and source['retrieved_at'] and source['search_trace_id']
    assert response.json()['web_citations'][0]['search_trace_id'] == source['search_trace_id']
    trace = TestClient(app).get('/traces/' + response.json()['trace_id']).json()
    last_model = [s for s in trace['spans'] if s['name'] == 'model.turn'][-1]
    assert last_model['attributes']['history_removed_evidence_copies'] == 1
    assert last_model['attributes']['history_saved_chars'] > 0
