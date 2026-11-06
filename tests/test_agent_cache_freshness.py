import json

from fastapi.testclient import TestClient
import httpx

from app.main import app
from tests.test_agent_loop_api import environment,message,call
from tests.test_search_cache import Clock
from tests.test_web_loop_api import search_response


def draft(value=None):
    return message(content=json.dumps({'claims':[] if value is None else [
        {'text':f'Probe值为{value}。','evidence_ids':['W1']}],
        'insufficient_evidence':value is None,'reason':'刷新失败，缺少当前证据。' if value is None else None}))


def provider(value):
    body=search_response()
    body['results'][0].update(title='Controlled freshness probe',url='https://example.org/freshness-probe',
        snippet=f'Current probe value is {value}.',summary=f'Controlled test source: probe value = {value}.')
    return body


def test_asgi_cache_keeps_snapshot_until_ttl_then_refreshes_and_never_serves_expired_on_error(environment,monkeypatch):
    from app import main
    from app.tools.search_cache import MemorySearchCache
    e=environment;clock=Clock()
    e.settings.web_search_enabled=True;e.settings.web_search_cache_enabled=True
    monkeypatch.setattr(main,'get_settings',lambda:e.settings)
    monkeypatch.setattr(main,'cache_from_settings',lambda settings:MemorySearchCache(ttl_seconds=10,clock=clock))
    def lookup():return message(calls=[call('w1','web_search',{'query':'controlled freshness probe','count':1})])
    e.answers.extend([lookup(),provider(1),draft(1),lookup(),draft(1),lookup(),provider(2),draft(2),
                      lookup(),httpx.ReadTimeout('private transport detail'),draft()])
    with TestClient(app) as http:
        def ask():
            response=http.post('/agent/loop',json={'prompt':'当前Probe值是多少？','enable_web_search':True})
            assert response.status_code==200,response.text
            return response.json()
        first=ask()
        clock.now=9
        within_ttl=ask()
        assert first['web_citations'][0]['summary']==within_ttl['web_citations'][0]['summary']
        assert first['web_citations'][0]['retrieved_at']==within_ttl['web_citations'][0]['retrieved_at']
        assert within_ttl['answer'].startswith('Probe值为1')
        clock.now=11
        refreshed=ask()
        assert refreshed['answer'].startswith('Probe值为2')
        assert refreshed['web_citations'][0]['summary']!=first['web_citations'][0]['summary']
        clock.now=22
        failed_refresh=ask()
        assert failed_refresh['insufficient_evidence'] is True and failed_refresh['web_citations']==[]
        assert 'private transport detail' not in failed_refresh['answer']
    assert len([r for r in e.requests if 'query' in r])==3
    assert all(client.is_closed for client in e.clients)
