import asyncio
import copy

import pytest
from pydantic import BaseModel, ConfigDict

from app.observability.store import InMemoryTraceStore


class Numbers(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    a: int
    b: int


def components(turns, **limits):
    from app.agent.loop_models import LoopLimits
    from app.agent.loop_runner import AgentLoopRunner
    from app.tools.async_registry import AsyncTool, AsyncToolRegistry
    registry = AsyncToolRegistry()
    executed = []
    async def add(args):
        executed.append((args.a, args.b))
        return args.a + args.b
    registry.register(AsyncTool('add_numbers', 'Add integers.', Numbers, add))
    histories = []
    async def model(messages, tools):
        histories.append(copy.deepcopy(messages))
        item = turns[len(histories)-1]
        if isinstance(item, BaseException):
            raise item
        return item
    traces = InMemoryTraceStore()
    runner = AgentLoopRunner(registry=registry, model=model, trace_store=traces, limits=LoopLimits(**limits))
    return runner, registry, executed, histories, traces


def turn(*calls, content=None):
    from app.agent.loop_models import LoopToolCall, LoopTurn
    return LoopTurn(content=content, calls=tuple(LoopToolCall(id=i, name=n, arguments=a) for i,n,a in calls))


def run(runner):
    return asyncio.run(runner.run('Compute the result.', request_id='request', trace_id='trace'))


def test_two_dependent_tool_rounds_then_final_answer_and_trace():
    runner, _, executed, histories, traces = components([
        turn(('c1', 'add_numbers', {'a':17,'b':25})),
        turn(('c2', 'add_numbers', {'a':42,'b':42})),
        turn(content='84'),
    ])
    result = run(runner)
    assert (result.status, result.answer, result.rounds, result.tool_calls) == ('completed','84',3,2)
    assert executed == [(17,25),(42,42)]
    import json
    assert json.loads(histories[1][-1]['content'])['output'] == 42
    assert histories[1][-1]['tool_call_id'] == 'c1'
    assert histories[2][-1]['tool_call_id'] == 'c2'
    trace = asyncio.run(traces.get('trace'))
    assert trace.status == 'ok'
    assert [s.name for s in trace.spans] == ['agent.loop','model.turn','tool.execute','model.turn','tool.execute','model.turn']
    assert trace.spans[0].attributes['rounds'] == 3
    assert all(s.parent_span_id == trace.spans[0].span_id for s in trace.spans[1:])
    assert [s.attributes['tool_name'] for s in trace.spans if s.name=='tool.execute']==['add_numbers','add_numbers']


def test_model_can_finish_without_tool_call():
    runner,_,executed,_,_ = components([turn(content='Hello')])
    result = run(runner)
    assert result.answer == 'Hello' and result.tool_calls == 0 and executed == []


def test_round_limit_stops_without_extra_summary_request():
    runner,_,executed,histories,traces = components([
        turn(('c1','add_numbers',{'a':1,'b':2})),
        turn(('c2','add_numbers',{'a':3,'b':4})),
    ],max_rounds=2)
    result = run(runner)
    assert result.status == 'limit_reached' and result.stop_reason == 'max_rounds'
    assert result.answer is None and len(histories) == 2 and len(executed) == 2
    trace=asyncio.run(traces.get('trace'))
    assert trace.spans[0].attributes['stop_reason'] == 'max_rounds'


def test_entire_batch_budget_checked_before_any_tool_execution():
    runner,_,executed,_,_=components([turn(
        ('c1','add_numbers',{'a':1,'b':2}),('c2','add_numbers',{'a':3,'b':4}),
    )],max_tool_calls=1)
    result=run(runner)
    assert result.stop_reason == 'max_tool_calls' and executed == [] and result.tool_calls == 0


@pytest.mark.parametrize('same_round',[True,False])
def test_duplicate_call_id_is_rejected_before_reexecution(same_round):
    from app.agent.loop_errors import LoopProtocolError
    call=('same','add_numbers',{'a':1,'b':2})
    turns=[turn(call,call)] if same_round else [turn(call),turn(call)]
    runner,_,executed,_,traces=components(turns)
    with pytest.raises(LoopProtocolError):
        run(runner)
    assert len(executed) == (0 if same_round else 1)
    assert asyncio.run(traces.get('trace')).status == 'error'


@pytest.mark.parametrize('call,code',[
    (('c1','unknown',{}),'unknown_tool'),
    (('c1','add_numbers',{'a':'17','b':25}),'invalid_arguments'),
    (('c1','add_numbers',{'a':17,'b':25,'extra':1}),'invalid_arguments'),
])
def test_safe_tool_error_can_be_corrected_in_next_round(call,code):
    import json
    runner,_,executed,histories,traces=components([turn(call),turn(('c2','add_numbers',{'a':17,'b':25})),turn(content='42')])
    result=run(runner)
    assert result.answer == '42' and executed == [(17,25)]
    error=json.loads(histories[1][-1]['content'])
    assert error == {'ok':False,'error':code,'attempts':0}
    assert any(s.status == 'error' for s in asyncio.run(traces.get('trace')).spans)


def test_tool_failure_is_sanitized_not_retried_and_model_can_report_failure():
    import json
    from app.tools.async_registry import AsyncTool
    runner,registry,_,histories,_=components([turn(('c1','broken',{'a':1,'b':2})),turn(content='Tool failed')])
    attempts=[]
    async def broken(args):
        attempts.append(1)
        raise RuntimeError('private remote body')
    registry.register(AsyncTool('broken','Broken',Numbers,broken,read_only=False))
    assert run(runner).answer == 'Tool failed'
    error=json.loads(histories[1][-1]['content'])
    assert error == {'ok':False,'error':'tool_failed','attempts':1}
    assert attempts == [1] and 'private' not in str(histories)


def test_only_explicit_transient_readonly_tool_is_retried():
    from app.tools.async_registry import AsyncTool, ToolTransientError
    runner,registry,_,_,traces=components([turn(('c1','retry',{'a':1,'b':2})),turn(content='3')])
    attempts=[]
    async def flaky(args):
        attempts.append(1)
        if len(attempts) == 1:
            raise ToolTransientError('retryable')
        return 3
    registry.register(AsyncTool('retry','Retry',Numbers,flaky,read_only=True,max_attempts=2))
    assert run(runner).tool_calls == 1 and len(attempts)==2
    trace=asyncio.run(traces.get('trace'))
    assert trace.spans[2].attributes['attempts'] == 2


def test_side_effecting_tool_cannot_enable_retries():
    from app.tools.async_registry import AsyncTool
    async def handler(args): return 1
    with pytest.raises(ValueError):
        AsyncTool('write','Write',Numbers,handler,read_only=False,max_attempts=2)


def test_tool_timeout_returns_error_without_retry():
    import json
    from app.tools.async_registry import AsyncTool
    runner,registry,_,histories,_=components([turn(('c1','slow',{'a':1,'b':2})),turn(content='Timed out')],tool_timeout_seconds=0.01)
    stopped=[]
    async def slow(args):
        try:
            await asyncio.Event().wait()
        finally:
            stopped.append(True)
    registry.register(AsyncTool('slow','Slow',Numbers,slow,read_only=True,max_attempts=2))
    assert run(runner).answer == 'Timed out'
    assert json.loads(histories[1][-1]['content'])['error'] == 'tool_timeout'
    assert stopped == [True]


@pytest.mark.parametrize('overall',[False,True])
def test_model_step_and_overall_timeout_are_distinct_terminations(overall):
    from app.agent.loop_errors import LoopTimeoutError
    runner,_,_,_,traces=components([],model_timeout_seconds=1 if overall else 0.01,
                                  run_timeout_seconds=0.01 if overall else 1)
    async def slow(*args): await asyncio.Event().wait()
    runner._model=slow
    with pytest.raises(LoopTimeoutError): run(runner)
    assert asyncio.run(traces.get('trace')).status == 'error'


def test_external_cancellation_propagates_and_is_traced():
    async def scenario():
        runner,_,_,_,traces=components([])
        entered=asyncio.Event()
        async def slow(*args):
            entered.set()
            await asyncio.Event().wait()
        runner._model=slow
        task=asyncio.create_task(runner.run('question',request_id='r',trace_id='t'))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError): await task
        assert (await traces.get('t')).status=='cancelled'
    asyncio.run(scenario())


def test_trace_write_failure_preserves_success_and_original_error():
    from app.agent.loop_errors import LoopProtocolError
    class Broken:
        async def save(self,trace): raise RuntimeError('trace down')
    runner,_,_,_,_=components([turn(content='ok')])
    runner._trace_store=Broken()
    assert run(runner).answer == 'ok'
    error=LoopProtocolError('invalid response')
    runner,_,_,_,_=components([error])
    runner._trace_store=Broken()
    with pytest.raises(LoopProtocolError) as caught: run(runner)
    assert caught.value is error


def test_context_budget_checked_before_next_model_request():
    from app.agent.loop_errors import LoopContextLimitError
    runner,_,_,histories,_=components([turn(content='ok')],max_history_chars=10)
    with pytest.raises(LoopContextLimitError): run(runner)
    assert histories == []


def test_oversize_tool_result_is_safe_error_not_silently_truncated():
    import json
    from app.tools.async_registry import AsyncTool
    runner,registry,_,histories,_=components([turn(('c1','huge',{'a':1,'b':2})),turn(content='Too large')],max_tool_result_chars=80)
    async def huge(args): return {'text':'x'*100}
    registry.register(AsyncTool('huge','Huge',Numbers,huge))
    assert run(runner).answer == 'Too large'
    assert json.loads(histories[1][-1]['content'])['error']=='tool_result_too_large'


@pytest.mark.parametrize('limits',[
    {'max_rounds':0},{'max_tool_calls':True},{'model_timeout_seconds':float('inf')},
    {'tool_timeout_seconds':0},{'run_timeout_seconds':-1},{'max_calls_per_turn':0},
])
def test_invalid_limits_rejected(limits):
    from app.agent.loop_models import LoopLimits
    with pytest.raises(ValueError): LoopLimits(**limits)


def test_too_many_calls_in_one_turn_rejected_before_execution():
    from app.agent.loop_errors import LoopProtocolError
    calls=tuple((f'c{i}','add_numbers',{'a':1,'b':2}) for i in range(5))
    runner,_,executed,_,_=components([turn(*calls)])
    with pytest.raises(LoopProtocolError): run(runner)
    assert executed==[]


def test_history_limit_stops_current_batch_before_next_tool_even_on_last_round():
    from app.agent.loop_errors import LoopContextLimitError
    from app.tools.async_registry import AsyncTool
    runner,registry,_,_,_=components([turn(('c1','large',{'a':1,'b':2}),('c2','large',{'a':3,'b':4}))],
        max_rounds=1,max_history_chars=1000)
    executed=[]
    async def large(args):
        executed.append(args.a)
        return 'x'*600
    registry.register(AsyncTool('large','Large',Numbers,large))
    with pytest.raises(LoopContextLimitError): run(runner)
    assert executed==[1]


def test_trace_persistence_has_bounded_cleanup_budget_preserving_business_result():
    runner,_,_,_,_=components([turn(content='ok')],trace_timeout_seconds=0.01)
    stopped=[]
    class Slow:
        async def save(self,trace):
            try:
                await asyncio.Event().wait()
            finally:
                stopped.append(True)
    runner._trace_store=Slow()
    assert run(runner).answer=='ok' and stopped==[True]
