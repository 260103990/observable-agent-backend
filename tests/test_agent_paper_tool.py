import asyncio
import json

import pytest
from pydantic import ValidationError

from tests.answer_helpers import hit


async def retrieve(tool, arguments):
    from app.agent.paper_tool import RetrievalInput
    from app.tools.async_registry import AsyncTool, AsyncToolRegistry
    registry=AsyncToolRegistry()
    registry.register(AsyncTool('paper_retrieval','Retrieve',RetrievalInput,tool.retrieve,read_only=True))
    result=await registry.execute('paper_retrieval',arguments.model_dump(),timeout_seconds=1,max_result_chars=32000)
    if not result['ok']:
        raise ValueError(result['error'])
    return result['output']


def make_tool(batches, **kwargs):
    from app.agent.paper_tool import PaperEvidenceTool
    calls=[]
    class Rag:
        async def retrieve(self,paper_id,query,**options):
            calls.append((paper_id,query,options))
            return batches[len(calls)-1]
    return PaperEvidenceTool(rag=Rag(),paper_id='paper-1',request_id='request',top_k=5,**kwargs),calls


def test_retrieval_scope_is_server_owned_and_ids_stable_across_rounds():
    from app.agent.paper_tool import RetrievalInput
    tool,calls=make_tool([[hit(0)],[hit(0),hit(1,text='A second result.')]])
    async def run():
        first=await retrieve(tool,RetrievalInput(query='First question'))
        second=await retrieve(tool,RetrievalInput(query='Rephrased question'))
        assert [e['evidence_id'] for e in first['evidence']]==['E1']
        assert [e['evidence_id'] for e in second['evidence']]==['E1','E2']
        assert all(e['page_number']==10 for e in second['evidence'])
    asyncio.run(run())
    assert [c[0] for c in calls]==['paper-1','paper-1']
    assert all(c[2]['top_k']==5 for c in calls)
    assert calls[0][2]['trace_id'] != calls[1][2]['trace_id']
    with pytest.raises(ValidationError): RetrievalInput(query='q',paper_id='other-paper')


def test_final_answer_uses_actual_evidence_and_unknown_reference_rejected():
    from app.agent.paper_tool import RetrievalInput
    from app.rag.evidence import CitationValidationError
    tool,_=make_tool([[hit()]])
    asyncio.run(retrieve(tool,RetrievalInput(query='baseline')))
    result=tool.finalize(json.dumps({'claims':[{'text':'Throughput is 42 requests/s.','evidence_ids':['E1']}],
                                   'insufficient_evidence':False,'reason':None}))
    assert '[E1]' in result.answer and result.citations[0].raw_text==hit().chunk.raw_text
    assert result.citations[0].page_number==10
    with pytest.raises(CitationValidationError):
        tool.finalize(json.dumps({'claims':[{'text':'A claim','evidence_ids':['E99']}],
                                 'insufficient_evidence':False,'reason':None}))


def test_no_retrieval_cannot_create_supported_paper_answer():
    from app.rag.evidence import CitationValidationError
    tool,_=make_tool([])
    with pytest.raises(CitationValidationError):
        tool.finalize(json.dumps({'claims':[{'text':'A claim','evidence_ids':['E1']}],
                                 'insufficient_evidence':False,'reason':None}))
    result=tool.finalize(json.dumps({'claims':[],'insufficient_evidence':True,'reason':'No evidence was retrieved.'}))
    assert result.insufficient_evidence and not result.citations


def test_evidence_budget_is_shared_across_tool_rounds():
    from app.agent.paper_tool import RetrievalInput
    tool,_=make_tool([[hit(0)],[hit(1)]],max_items=1)
    async def run():
        await retrieve(tool,RetrievalInput(query='first'))
        second=await retrieve(tool,RetrievalInput(query='second'))
        assert len(second['evidence'])==1 and second['omitted_count']==1
        result=tool.finalize(json.dumps({'claims':[{'text':'Supported','evidence_ids':['E1']}],
                                        'insufficient_evidence':False,'reason':None}))
        assert result.omitted_evidence_count==1
    asyncio.run(run())


def test_foreign_paper_and_conflicting_source_are_rejected():
    from app.agent.paper_tool import RetrievalInput
    tool,_=make_tool([[hit(paper_id='foreign')]])
    with pytest.raises(ValueError): asyncio.run(retrieve(tool,RetrievalInput(query='q')))
    tool,_=make_tool([[hit(0)],[hit(0,text='Changed source')]])
    asyncio.run(retrieve(tool,RetrievalInput(query='q')))
    with pytest.raises(ValueError): asyncio.run(retrieve(tool,RetrievalInput(query='q again')))


def test_oversize_tool_payload_does_not_authorize_unseen_evidence():
    from app.agent.paper_tool import RetrievalInput
    from app.rag.evidence import CitationValidationError
    tool,_=make_tool([[hit()]],max_result_chars=10)
    with pytest.raises(ValueError): asyncio.run(retrieve(tool,RetrievalInput(query='q')))
    with pytest.raises(CitationValidationError):
        tool.finalize(json.dumps({'claims':[{'text':'A claim','evidence_ids':['E1']}],
                                 'insufficient_evidence':False,'reason':None}))


@pytest.mark.parametrize('content',['not JSON','[]','{"claims":[],"claims":[]}'])
def test_invalid_final_json_rejected(content):
    from app.rag.evidence import CitationValidationError
    tool,_=make_tool([])
    with pytest.raises(CitationValidationError): tool.finalize(content)


def test_actual_executor_budget_cannot_authorize_a_rejected_tool_result():
    from app.agent.paper_tool import RetrievalInput
    from app.tools.async_registry import AsyncTool, AsyncToolRegistry
    from app.rag.evidence import CitationValidationError
    tool,_=make_tool([[hit()]])
    registry=AsyncToolRegistry()
    registry.register(AsyncTool('paper_retrieval','Retrieve',RetrievalInput,tool.retrieve,read_only=True))
    result=asyncio.run(registry.execute('paper_retrieval',{'query':'q'},timeout_seconds=1,max_result_chars=100))
    assert result['error']=='tool_result_too_large'
    with pytest.raises(CitationValidationError):
        tool.finalize(json.dumps({'claims':[{'text':'A claim','evidence_ids':['E1']}],
                                 'insufficient_evidence':False,'reason':None}))
