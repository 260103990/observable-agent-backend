"""Real test PostgreSQL + new HTTP loop; model and embedding HTTP are controlled."""
import asyncio
import json
import os
from pathlib import Path

import asyncpg
from fastapi.testclient import TestClient
import httpx
import pytest

from app.config import get_settings
from app.main import app
from app.rag import pdf
from tests.pdf_helpers import write_text_pdf


pytestmark=pytest.mark.postgres
ROOT=Path(pdf.__file__).parents[2]


def test_real_database_import_two_retrieval_rounds_and_citation(monkeypatch,tmp_path):
    url=os.getenv('TEST_PGVECTOR_DATABASE_URL')
    if not url:
        pytest.skip('TEST_PGVECTOR_DATABASE_URL is not configured')
    if not url.rsplit('/',1)[-1].split('?',1)[0].endswith('_test'):
        pytest.fail('This test only accepts a database name ending in _test')
    async def initialize():
        connection=await asyncpg.connect(url)
        try:
            await connection.execute((ROOT/'app/rag/schema.sql').read_text(encoding='utf-8'))
            await connection.execute((ROOT/'app/observability/schema.sql').read_text(encoding='utf-8'))
        finally:
            await connection.close()
    asyncio.run(initialize())
    papers,traces,embedding_inputs=[],set(),[]
    original=httpx.AsyncClient
    def handler(request):
        if request.url.path=='/api/tags':
            return httpx.Response(200,json={'models':[{'name':'qwen3-embedding:0.6b','digest':'v2-test-revision'}]})
        body=json.loads(request.content)
        if request.url.path=='/api/embed':
            embedding_inputs.extend(body['input'])
            return httpx.Response(200,json={'model':'qwen3-embedding:0.6b','embeddings':[[1.0]+[0.0]*1023 for _ in body['input']]})
        tool_results=[json.loads(m['content']) for m in body['messages'] if m['role']=='tool']
        for result in tool_results:
            if result.get('ok'):
                traces.add(result['output']['retrieval_trace_id'])
        if len(tool_results)<2:
            message={'role':'assistant','content':None,'tool_calls':[{'id':f'retrieval-{len(tool_results)}',
                'type':'function','function':{'name':'paper_retrieval',
                'arguments':json.dumps({'query':'baseline throughput' if not tool_results else 'baseline requests per second'})}}]}
            finish='tool_calls'
        else:
            message={'role':'assistant','content':json.dumps({'claims':[{'text':'Baseline throughput is 42 requests/s.',
                'evidence_ids':['E1']}],'insufficient_evidence':False,'reason':None})}
            finish='stop'
        return httpx.Response(200,json={'id':'controlled-loop','model':'deepseek-flash',
            'choices':[{'finish_reason':finish,'message':message}]})
    monkeypatch.setattr(httpx,'AsyncClient',lambda **kwargs:original(transport=httpx.MockTransport(handler),**kwargs))
    monkeypatch.setenv('DATABASE_URL',url)
    monkeypatch.setenv('RAG_ENABLED','true')
    monkeypatch.setenv('RAG_STORAGE_DIR',str(tmp_path/'pdfs'))
    monkeypatch.setenv('DEEPSEEK_API_KEY','controlled-integration-key')
    get_settings.cache_clear()
    fixture=tmp_path/'paper.pdf'
    write_text_pdf(fixture,['Baseline throughput is 42 requests/s.'])
    try:
        with TestClient(app) as client:
            response=client.post('/papers',files={'file':('paper.pdf',fixture.read_bytes(),'application/pdf')})
            assert response.status_code==201,response.text
            imported=response.json()
            papers.append(imported['paper_id'])
            traces.add(imported['trace_id'])
            response=client.post('/agent/loop',json={'paper_id':imported['paper_id'],'prompt':'What is the baseline throughput?'})
            assert response.status_code==200,response.text
            result=response.json()
            traces.add(result['trace_id'])
            assert result['status']=='completed' and result['rounds']==3 and result['tool_calls']==2
            assert result['citations'][0]['paper_id']==imported['paper_id']
            assert result['citations'][0]['page_number']==1
            assert result['citations'][0]['raw_text']=='Baseline throughput is 42 requests/s.'
            trace=client.get('/traces/'+result['trace_id']).json()
            assert trace['status']=='ok' and trace['spans'][0]['name']=='agent.loop'
            assert len(embedding_inputs)==3  # one document, two queries
            assert len(traces)==4  # import, loop, and two linked retrieval traces
    finally:
        get_settings.cache_clear()
        async def cleanup():
            connection=await asyncpg.connect(url)
            try:
                await connection.execute('DELETE FROM rag_papers WHERE paper_id=ANY($1::text[])',papers)
                await connection.execute('DELETE FROM traces WHERE trace_id=ANY($1::text[])',list(traces))
            finally:
                await connection.close()
        asyncio.run(cleanup())
