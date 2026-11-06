"""Opt-in real external Search and Agent tests; never select a database."""
import asyncio
import json
import os
from pathlib import Path

from fastapi.testclient import TestClient
import httpx
import pytest

from app.config import get_settings
from app.main import app
from app.observability.store import InMemoryTraceStore
from app.tools.web_search import DeepSeekSearchClient


pytestmark=pytest.mark.postgres  # Existing opt-in integration selection; these cases need no DB.


def settings():
    if os.getenv('RUN_WEB_SEARCH_LIVE_TESTS')!='1':
        pytest.skip('Opt in with RUN_WEB_SEARCH_LIVE_TESTS=1')
    value=get_settings().model_copy(update={'database_url':None,'rag_enabled':False,'web_search_enabled':True})
    if value.deepseek_api_key is None:
        pytest.skip('Configure the authorized key locally')
    return value


def record(name,value):
    output=os.getenv('WEB_SEARCH_LIVE_REPORT_PATH')
    if output:
        path=Path(output)
        content=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        content[name]=value
        path.write_text(json.dumps(content,ensure_ascii=False,indent=2),encoding='utf-8')


def test_real_deepseek_search_api():
    value=settings()
    async def run():
        traces=InMemoryTraceStore()
        async with httpx.AsyncClient(trust_env=False) as http:
            result=await DeepSeekSearchClient(client=http,api_key=value.deepseek_api_key,trace_store=traces).search(
                'site:api-docs.deepseek.com tool calls function calling',count=2,request_id='live-direct-search',trace_id='live-search')
        assert result.results and all(r.url.startswith('https://') for r in result.results)
        trace=await traces.get('live-search')
        assert trace.status=='ok' and trace.spans[0].attributes['outbound_request_attempts']==1
        record('direct_search',{'batch':result.model_dump(),'trace':trace.model_dump()})
    asyncio.run(run())


def test_real_deepseek_agent_web_search(monkeypatch):
    from app import main
    value=settings()
    monkeypatch.setattr(main,'get_settings',lambda:value)
    app.dependency_overrides[get_settings]=lambda:value
    try:
        with TestClient(app) as client:
            response=client.post('/agent/loop',json={'enable_web_search':True,'prompt':
                '请使用 web_search 查找 DeepSeek 官方 Tool Calls 文档，简要说明 function calling 中是谁执行实际函数，并给出来源。查询限定 site:api-docs.deepseek.com。'})
            assert response.status_code==200,response.text
            result=response.json()
            assert result['status']=='completed' and result['tool_calls']>=1 and result['web_citations']
            assert result['citations']==[]
            trace=client.get('/traces/'+result['trace_id']).json()
            children={item['search_trace_id']:client.get('/traces/'+item['search_trace_id']).json()
                      for item in result['web_citations']}
            assert trace['status']=='ok' and all(t['status']=='ok' for t in children.values())
            record('agent_web',{'result':result,'trace':trace,'cited_search_traces':children})
    finally:
        app.dependency_overrides.pop(get_settings,None)
        get_settings.cache_clear()
