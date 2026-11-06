"""Opt-in real DeepSeek, plus real Ollama/pgvector for the paper case."""
import asyncio
import json
import os
from pathlib import Path

import asyncpg
from fastapi.testclient import TestClient
import pytest

from app.config import get_settings
from app.main import app


pytestmark=pytest.mark.postgres


def enabled():
    if os.getenv('RUN_AGENT_LOOP_LIVE_TESTS')!='1':
        pytest.skip('Opt in with RUN_AGENT_LOOP_LIVE_TESTS=1')
    if get_settings().deepseek_api_key is None:
        pytest.skip('Configure DEEPSEEK_API_KEY locally')


def record(name,value):
    output=os.getenv('AGENT_LOOP_LIVE_REPORT_PATH')
    if output:
        path=Path(output)
        data=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        data[name]=value
        path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def test_real_deepseek_dependent_calculation(monkeypatch):
    enabled()
    from app import main
    # Preserve local model credentials while explicitly disabling both env and dotenv DB selection.
    settings=get_settings().model_copy(update={'database_url':None,'rag_enabled':False})
    monkeypatch.setattr(main,'get_settings',lambda:settings)
    app.dependency_overrides[get_settings]=lambda:settings
    monkeypatch.delenv('DATABASE_URL',raising=False)
    monkeypatch.setenv('RAG_ENABLED','false')
    get_settings.cache_clear()
    try:
        with TestClient(app) as client:
            response=client.post('/agent/loop',json={'prompt':
                '请先调用 add_numbers 计算 17+25，再用工具将这个结果加上 42，最后简洁回答最终数字。必须分两次调用工具。'})
            assert response.status_code==200,response.text
            result=response.json()
            assert result['status']=='completed' and '84' in result['answer'] and result['tool_calls']>=2
            trace=client.get('/traces/'+result['trace_id']).json()
            assert trace['status']=='ok'
            record('calculation',{'result':result,'trace_span_names':[s['name'] for s in trace['spans']]})
    finally:
        app.dependency_overrides.pop(get_settings,None)
        get_settings.cache_clear()


def test_real_deepseek_ollama_postgres_paper_loop(monkeypatch,tmp_path):
    enabled()
    url=os.getenv('TEST_PGVECTOR_DATABASE_URL')
    source=os.getenv('PAPER_PDF_PATH')
    if not url or not source:
        pytest.skip('Configure isolated test DB and PAPER_PDF_PATH')
    if not url.rsplit('/',1)[-1].split('?',1)[0].endswith('_test'):
        pytest.fail('Live test requires a database ending in _test')
    monkeypatch.setenv('DATABASE_URL',url)
    monkeypatch.setenv('RAG_ENABLED','true')
    monkeypatch.setenv('RAG_STORAGE_DIR',str(tmp_path/'pdfs'))
    get_settings.cache_clear()
    papers,traces=[],set()
    try:
        with TestClient(app) as client:
            response=client.post('/papers',files={'file':('cortex.pdf',Path(source).read_bytes(),'application/pdf')})
            assert response.status_code==201,response.text
            imported=response.json()
            papers.append(imported['paper_id'])
            traces.add(imported['trace_id'])
            response=client.post('/agent/loop',json={'paper_id':imported['paper_id'],'top_k':5,
                'prompt':'请通过 paper_retrieval 查找实验中采用的语言模型，并说明模型名称。必须给出论文证据；找不到的信息明确说明不足。'})
            assert response.status_code==200,response.text
            result=response.json()
            traces.add(result['trace_id'])
            assert result['status']=='completed' and result['tool_calls']>=1 and result['citations']
            assert all(c['paper_id']==imported['paper_id'] and c['raw_text'] for c in result['citations'])
            trace=client.get('/traces/'+result['trace_id']).json()
            assert trace['status']=='ok'
            record('paper',{'imported':imported,'result':result,'trace_span_names':[s['name'] for s in trace['spans']]})
    finally:
        get_settings.cache_clear()
        # Child retrieval traces share the unique loop request_id; delete only this run's rows.
        async def cleanup():
            connection=await asyncpg.connect(url)
            try:
                request_ids=await connection.fetch('SELECT request_id FROM traces WHERE trace_id=ANY($1::text[])',list(traces))
                await connection.execute('DELETE FROM rag_papers WHERE paper_id=ANY($1::text[])',papers)
                await connection.execute('DELETE FROM traces WHERE request_id=ANY($1::text[])',[r['request_id'] for r in request_ids])
            finally:
                await connection.close()
        asyncio.run(cleanup())
