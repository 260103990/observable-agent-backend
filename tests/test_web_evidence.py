import asyncio
import json

import pytest

from app.tools.async_registry import AsyncTool, AsyncToolRegistry
from tests.answer_helpers import hit


def batch(query='q',summary='Provider search summary.',url='https://example.org/a',trace_id='search-1'):
    from app.tools.web_models import SearchBatch, SearchResult
    return SearchBatch(query=query,retrieved_at='2026-11-04T00:00:00+00:00',trace_id=trace_id,response_body_bytes=123,
        results=(SearchResult(rank=1,title='Search title',url=url,snippet='Snippet',summary=summary,
                              rerank_score=2.0,authority=3),))


def make_tool(batches,**options):
    from app.agent.web_evidence import WebEvidenceTool
    seen=[]
    class Search:
        async def search(self,query,**kwargs):
            seen.append((query,kwargs))
            return batches[len(seen)-1]
    return WebEvidenceTool(search=Search(),request_id='request',**options),seen


async def invoke(tool,query='q',max_result_chars=32000):
    from app.tools.web_models import SearchInput
    registry=AsyncToolRegistry()
    registry.register(AsyncTool('web_search','Search',SearchInput,tool.search,read_only=True))
    return await registry.execute('web_search',{'query':query,'count':5},timeout_seconds=1,max_result_chars=max_result_chars)


def draft(ids=('W1',),text='A supported result.'):
    return json.dumps({'claims':[{'text':text,'evidence_ids':list(ids)}],'insufficient_evidence':False,'reason':None})


def test_web_ids_stable_and_search_summary_provenance_preserved():
    tool,seen=make_tool([batch(),batch(query='q2',url='https://example.org/b',trace_id='search-2')])
    first=asyncio.run(invoke(tool))
    second=asyncio.run(invoke(tool,'q2'))
    assert [e['evidence_id'] for e in first['output']['evidence']]==['W1']
    assert [e['evidence_id'] for e in second['output']['evidence']]==['W1','W2']
    item=tool.evidence[0]
    assert item.source_type=='web_search_result' and item.summary=='Provider search summary.'
    assert item.query=='q' and item.search_trace_id=='search-1' and item.retrieved_at
    assert 'raw_text' not in item.model_dump() and 'page_number' not in item.model_dump()
    assert seen[0][1]['trace_id']!=seen[1][1]['trace_id']


def test_same_url_query_dependent_summaries_are_distinct_but_exact_duplicates_deduplicate():
    tool,_=make_tool([batch(),batch(),batch(query='different',summary='Another summary.')])
    asyncio.run(invoke(tool));asyncio.run(invoke(tool));asyncio.run(invoke(tool,'different'))
    assert len(tool.evidence)==2
    assert [e.evidence_id for e in tool.evidence]==['W1','W2']


def test_budget_and_empty_summary_are_reported_without_truncating_provider_text():
    tool,_=make_tool([batch(),batch(url='https://example.org/b')],max_items=1)
    asyncio.run(invoke(tool))
    response=asyncio.run(invoke(tool))
    assert len(tool.evidence)==1 and response['output']['omitted_count']==1
    assert tool.evidence[0].summary=='Provider search summary.'
    too_small,_=make_tool([batch()],max_chars=1)
    response=asyncio.run(invoke(too_small))
    assert response['output']['evidence']==[] and response['output']['omitted_count']==1


def test_rejected_executor_payload_does_not_authorize_web_reference():
    from app.agent.web_evidence import GroundedWebAnswer
    from app.rag.evidence import CitationValidationError
    tool,_=make_tool([batch()])
    response=asyncio.run(invoke(tool,max_result_chars=100))
    assert response['error']=='tool_result_too_large' and tool.evidence==()
    with pytest.raises(CitationValidationError): GroundedWebAnswer(tool).finalize(draft())


def test_final_web_answer_renders_backend_owned_url_and_snapshot():
    from app.agent.web_evidence import GroundedWebAnswer
    tool,_=make_tool([batch()])
    asyncio.run(invoke(tool))
    answer=GroundedWebAnswer(tool).finalize(draft())
    assert answer.answer.endswith('[W1]') and answer.web_citations[0].url=='https://example.org/a'
    assert answer.web_citations[0].summary=='Provider search summary.' and answer.citations==()


def test_mixed_paper_and_web_ids_have_separate_citation_types():
    from app.agent.paper_tool import PaperEvidenceTool, RetrievalInput
    from app.agent.web_evidence import GroundedWebAnswer
    class Rag:
        async def retrieve(self,*args,**kwargs): return [hit()]
    paper=PaperEvidenceTool(rag=Rag(),paper_id='paper-1',request_id='request')
    registry=AsyncToolRegistry()
    registry.register(AsyncTool('paper_retrieval','Retrieve',RetrievalInput,paper.retrieve,read_only=True))
    asyncio.run(registry.execute('paper_retrieval',{'query':'q'},timeout_seconds=1,max_result_chars=32000))
    web,_=make_tool([batch()])
    asyncio.run(invoke(web))
    answer=GroundedWebAnswer(web,paper=paper).finalize(draft(('E1','W1')))
    assert '[E1][W1]' in answer.answer
    assert answer.citations[0].page_number==10 and answer.web_citations[0].url=='https://example.org/a'


@pytest.mark.parametrize('content',[
    draft(('W99',)),draft(('E1',)),draft(text='Claim [W1]'),draft(text='Claim [E99]'),
    draft(text='Visit https://invented.example/a'), 'not JSON', '[]',
    json.dumps({'claims':[],'insufficient_evidence':False,'reason':None}),
])
def test_unknown_citations_forged_markers_urls_and_bad_structure_rejected(content):
    from app.agent.web_evidence import GroundedWebAnswer
    from app.rag.evidence import CitationValidationError
    tool,_=make_tool([batch()]);asyncio.run(invoke(tool))
    with pytest.raises(CitationValidationError): GroundedWebAnswer(tool).finalize(content)


def test_no_evidence_can_return_explicit_insufficiency():
    from app.agent.web_evidence import GroundedWebAnswer
    tool,_=make_tool([])
    answer=GroundedWebAnswer(tool).finalize(json.dumps({'claims':[],'insufficient_evidence':True,'reason':'No search evidence.'}))
    assert answer.insufficient_evidence and not answer.web_citations


def test_omitted_web_count_reaches_final_response():
    from app.agent.web_evidence import GroundedWebAnswer
    tool,_=make_tool([batch(),batch(url='https://example.org/b')],max_items=1)
    asyncio.run(invoke(tool));asyncio.run(invoke(tool))
    assert GroundedWebAnswer(tool).finalize(draft()).omitted_web_evidence_count==1


@pytest.mark.parametrize('text',[
    'See [source](//invented.example/source).',
    'See [source](https&#58;//invented.example/source).',
    'See [nested [source]](//invented.example).',
    'See <a href="//invented.example">source</a>.',
    '[ref]: //invented.example/source',
])
@pytest.mark.parametrize('in_reason',[False,True])
def test_model_generated_markdown_or_html_link_cannot_bypass_source_ownership(text,in_reason):
    from app.agent.web_evidence import GroundedWebAnswer
    from app.rag.evidence import CitationValidationError
    tool,_=make_tool([batch()]);asyncio.run(invoke(tool))
    content=json.dumps({'claims':[],'insufficient_evidence':True,'reason':text}) if in_reason else draft(text=text)
    with pytest.raises(CitationValidationError): GroundedWebAnswer(tool).finalize(content)
