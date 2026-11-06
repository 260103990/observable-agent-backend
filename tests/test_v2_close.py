import asyncio
import json

import httpx
import pytest

from app.config import Settings
from app.rag.evidence import CitationValidationError
from tests.test_agent_loop import components,turn,run
from tests.test_agent_paper_tool import make_tool,retrieve
from tests.test_web_evidence import make_tool as make_web_tool,invoke,batch
from tests.test_web_search import execute,result


GOOD=json.dumps({'claims':[],'insufficient_evidence':True,'reason':'No supporting evidence was retrieved.'})
BAD=json.dumps({'claims':[],'insufficient_evidence':False,'reason':'No supporting evidence was retrieved.'})


@pytest.mark.parametrize('wrapper',[
    lambda s:s,lambda s:'```json\n'+s+'\n```',lambda s:'```\n'+s+'\n```',lambda s:' \r\n```JSON\r\n'+s+'\r\n```\r\n ',
])
def test_only_complete_final_json_fences_are_normalized(wrapper):
    from app.agent.final_output import load_final_json_object
    assert load_final_json_object(wrapper(GOOD))==json.loads(GOOD)


@pytest.mark.parametrize('content',[
    'Before\n```json\n{}\n```','```json\n{}\n```\nAfter','```python\n{}\n```',
    '```json\n{}\n```\n```json\n{}\n```','```json\n{}','```json {} ```',
    '```json\n[]\n```','```json\n{"a":1,"a":2}\n```','```json\n{"a":NaN}\n```',
])
def test_final_normalization_never_extracts_arbitrary_or_invalid_json(content):
    from app.agent.final_output import load_final_json_object
    with pytest.raises(ValueError):load_final_json_object(content)


def test_tool_arguments_remain_strict_json_without_markdown_normalization():
    from app.agent.loop_models import load_json_object
    with pytest.raises(ValueError):load_json_object('```json\n{"a":1}\n```')


def test_paper_and_web_finalizers_accept_fences_but_still_validate_references():
    from app.agent.paper_tool import PaperEvidenceTool
    from app.agent.web_evidence import GroundedWebAnswer
    paper=PaperEvidenceTool(rag=None,paper_id='paper-1',request_id='r')
    web,_=make_web_tool([])
    for finalize in (paper.finalize,GroundedWebAnswer(web).finalize):
        answer=finalize('```json\n'+GOOD+'\n```')
        assert answer.insufficient_evidence and answer.citations==()
        unknown=json.dumps({'claims':[{'text':'A claim','evidence_ids':['E99']}],'insufficient_evidence':False,'reason':None})
        with pytest.raises(CitationValidationError):finalize('```json\n'+unknown+'\n```')
        with pytest.raises(CitationValidationError):finalize('```json\n'+BAD+'\n```')


@pytest.mark.parametrize('title,summary,snippet',[('', 'x'*6000,'normal'),('   ','normal','x'*3000),('x'*600,'normal','')],
                         ids=['empty-title-long-summary','blank-title-long-snippet','long-title'])
def test_search_preserves_empty_title_and_long_text_within_total_response_budget(title,summary,snippet):
    action,_,_=execute(httpx.Response(200,json={'results':[result(title=title,summary=summary,snippet=snippet)]}))
    item=asyncio.run(action()).results[0]
    assert item.title==title and item.summary==summary and item.snippet==snippet


def test_overbudget_summary_is_omitted_whole_without_losing_other_search_results():
    from app.tools.web_models import SearchBatch,SearchResult
    values=(SearchResult(rank=1,title='',url='https://example.org/large',snippet='',summary='x'*14000,rerank_score=2.0,authority=3),
            SearchResult(rank=2,title='',url='https://example.org/small',snippet='Short evidence.',summary='Supported summary.',rerank_score=2.0,authority=3))
    response=SearchBatch(query='q',retrieved_at='2026-11-04T00:00:00+00:00',trace_id='search',response_body_bytes=15000,results=values)
    tool,_=make_web_tool([response])
    observed=asyncio.run(invoke(tool))
    assert observed['ok'] and observed['output']['omitted_count']==1
    assert len(tool.evidence)==1 and tool.evidence[0].url=='https://example.org/small'
    assert tool.evidence[0].title=='' and tool.evidence[0].summary=='Supported summary.'


def empty_paper():
    from app.agent.paper_tool import PaperEvidenceTool
    return PaperEvidenceTool(rag=None,paper_id='paper-1',request_id='r')


def test_one_final_repair_uses_existing_rounds_and_does_not_repeat_tools():
    runner,_,executed,histories,traces=components([
        turn(('c1','add_numbers',{'a':1,'b':2})),turn(content=BAD),turn(content=GOOD)],max_rounds=3)
    runner._finalizer=empty_paper().finalize
    answer=run(runner)
    assert answer.status=='completed' and answer.rounds==3 and answer.final_repairs==1
    assert answer.tool_calls==1 and executed==[(1,2)]
    assert histories[2][-2]=={'role':'assistant','content':BAD}
    assert histories[2][-1]['role']=='system' and json.loads(histories[2][-2]['content'])['insufficient_evidence'] is False
    trace=asyncio.run(traces.get('trace'))
    assert trace.spans[0].attributes['final_repairs']==1
    assert [s.status for s in trace.spans if s.name=='answer.validate']==['error','ok']


@pytest.mark.parametrize('limits',[{'max_final_repairs':0},{'max_rounds':1}])
def test_disabled_or_exhausted_budget_never_starts_extra_repair_request(limits):
    runner,_,_,histories,traces=components([turn(content=BAD)],**limits)
    runner._finalizer=empty_paper().finalize
    with pytest.raises(CitationValidationError):run(runner)
    assert len(histories)==1
    assert asyncio.run(traces.get('trace')).spans[0].attributes['final_repairs']==0


def test_second_invalid_final_is_rejected_without_silent_state_flip():
    runner,_,_,histories,traces=components([turn(content=BAD),turn(content=BAD)])
    runner._finalizer=empty_paper().finalize
    with pytest.raises(CitationValidationError):run(runner)
    assert len(histories)==2 and asyncio.run(traces.get('trace')).status=='error'


def test_model_cannot_execute_tools_during_final_answer_repair():
    from app.agent.loop_errors import LoopProtocolError
    runner,_,executed,_,traces=components([turn(content=BAD),turn(('c1','add_numbers',{'a':1,'b':2}))])
    runner._finalizer=empty_paper().finalize
    with pytest.raises(LoopProtocolError):run(runner)
    assert executed==[] and not [s for s in asyncio.run(traces.get('trace')).spans if s.name=='tool.execute']


def test_repair_never_catches_unrelated_finalizer_programming_errors():
    runner,_,_,histories,_=components([turn(content='text')])
    error=RuntimeError('original implementation failure')
    def broken(content):raise error
    runner._finalizer=broken
    with pytest.raises(RuntimeError) as caught:run(runner)
    assert caught.value is error and len(histories)==1


@pytest.mark.parametrize('overall',[False,True])
def test_repair_preserves_model_timeout_and_global_deadline(overall):
    from app.agent.loop_errors import LoopTimeoutError
    runner,_,_,_,traces=components([],model_timeout_seconds=1 if overall else 0.01,
                                  run_timeout_seconds=0.01 if overall else 1)
    calls=[]
    async def model(*args):
        calls.append(1)
        if len(calls)==1:return turn(content=BAD)
        await asyncio.Event().wait()
    runner._model=model;runner._finalizer=empty_paper().finalize
    with pytest.raises(LoopTimeoutError):run(runner)
    assert len(calls)==2 and asyncio.run(traces.get('trace')).spans[0].attributes['final_repairs']==1


def test_external_cancellation_during_repair_is_not_swallowed():
    async def scenario():
        runner,_,_,_,traces=components([])
        entered=asyncio.Event();calls=[]
        async def model(*args):
            calls.append(1)
            if len(calls)==1:return turn(content=BAD)
            entered.set();await asyncio.Event().wait()
        runner._model=model;runner._finalizer=empty_paper().finalize
        task=asyncio.create_task(runner.run('q',request_id='r',trace_id='t'))
        try:
            await asyncio.wait_for(entered.wait(),timeout=1)
        except TimeoutError:
            if task.done():await task
            task.cancel();await asyncio.gather(task,return_exceptions=True)
            raise
        task.cancel()
        with pytest.raises(asyncio.CancelledError):await task
        assert (await traces.get('t')).status=='cancelled'
    asyncio.run(scenario())


def test_repair_configuration_and_response_defaults_are_bounded():
    from app.agent.loop_models import LoopLimits
    assert Settings(_env_file=None).agent_loop_max_final_repairs==1
    assert LoopLimits().max_final_repairs==1
    for value in (-1,2,True):
        with pytest.raises(ValueError):LoopLimits(max_final_repairs=value)


def test_repair_rejected_by_history_budget_is_not_counted_as_a_model_attempt():
    from app.agent.loop_errors import LoopContextLimitError
    runner,_,_,histories,traces=components([turn(content=BAD)],max_history_chars=800)
    runner._finalizer=empty_paper().finalize
    with pytest.raises(LoopContextLimitError):run(runner)
    trace=asyncio.run(traces.get('trace'))
    assert len(histories)==1 and trace.spans[0].attributes['final_repairs']==0
