import asyncio
import copy
import json

import httpx
import pytest


def exchange(identifier, items, name='paper_retrieval', *, ok=True):
    result = {'ok': True, 'output': {'evidence': items, 'retrieval_trace_id': identifier}, 'attempts': 1}
    if not ok:
        result = {'ok': False, 'error': 'tool_failed', 'attempts': 1}
    return [
        {'role': 'assistant', 'content': None, 'tool_calls': [{'id': identifier, 'type': 'function',
         'function': {'name': name, 'arguments': '{"query":"q"}'}}]},
        {'role': 'tool', 'tool_call_id': identifier, 'content': json.dumps(result, ensure_ascii=False)},
    ]


def evidence(identifier='E1', text=None):
    return {'evidence_id': identifier, 'page_number': 10, 'text': text or 'Source passage. ' * 80}


def project(messages, tools=('paper_retrieval', 'web_search')):
    from app.agent.evidence_history import project_history
    return project_history(messages, evidence_tools=tools)


def test_duplicate_snapshots_point_directly_to_latest_full_copy_and_restore_exactly():
    from app.agent.evidence_history import restore_evidence_history
    source = exchange('c1', [evidence()]) + exchange('c2', [evidence(), evidence('E2')]) + exchange('c3', [evidence(), evidence('E2')])
    original = copy.deepcopy(source)
    view = project(source)
    assert view.removed_evidence_copies == 3 and view.chars < view.raw_chars
    for index in (1, 3):
        for item in json.loads(view.messages[index]['content'])['output']['evidence']:
            assert item['same_as_tool_call_id'] == 'c3'
    assert restore_evidence_history(view.messages) == original
    assert source == original
    view.messages[0]['content'] = 'changed copy'
    assert source == original


@pytest.mark.parametrize('change', ['text', 'page_number', 'evidence_id', 'source_time'])
def test_different_snapshot_fields_are_not_merged(change):
    first = evidence()
    second = dict(first)
    if change == 'text': second['text'] = 'Changed paragraph. ' * 80
    if change == 'page_number': second['page_number'] = 11
    if change == 'evidence_id': second['evidence_id'] = 'E2'
    if change == 'source_time': second['retrieved_at'] = 'later'
    source = exchange('c1', [first]) + exchange('c2', [second])
    view = project(source)
    assert view.removed_evidence_copies == 0 and view.messages == source


def test_only_explicit_owned_tool_names_are_eligible_and_failures_stay_visible():
    source = exchange('c1', [evidence()], 'custom') + exchange('c2', [evidence()], 'custom')
    assert project(source).messages == source
    source = exchange('p1', [evidence()]) + exchange('p2', [], ok=False)
    assert project(source).messages == source
    assert project(exchange('p1', [evidence()]) + exchange('p2', [evidence()]), tools=()).removed_evidence_copies == 0


def test_paper_web_namespaces_and_malformed_items_are_never_conflated():
    source = exchange('p1', [evidence()]) + exchange('w1', [evidence('W1')], 'web_search')
    assert project(source).removed_evidence_copies == 0
    for items in ([evidence(), evidence()], [{'text': 'No ID'}], ['not an item']):
        source = exchange('p1', items) + exchange('p2', items)
        assert project(source).messages == source


def test_short_evidence_is_not_replaced_by_a_longer_reference():
    source = exchange('c1', [{'evidence_id': 'E1'}]) + exchange('c2', [{'evidence_id': 'E1'}])
    assert project(source).messages == source


@pytest.mark.parametrize('target,name', [('missing', 'paper_retrieval'), ('c2', 'web_search')])
def test_restore_rejects_dangling_or_cross_tool_reference(target, name):
    from app.agent.evidence_history import restore_evidence_history
    source = exchange('c1', [{'evidence_id': 'E1', 'same_as_tool_call_id': target}])
    source += exchange('c2', [evidence()], name)
    with pytest.raises(ValueError): restore_evidence_history(source)


def test_evaluation_observes_original_results_after_projection_and_detects_real_conflict():
    from app.evaluation.recording import EvaluationCapture
    capture = EvaluationCapture(model_url='https://model.local')
    first = exchange('c1', [evidence()])
    both = first + exchange('c2', [evidence()])
    async def scenario():
        with capture.for_request('r'):
            for messages in (first, project(both).messages):
                await capture.on_request(httpx.Request('POST', 'https://model.local', json={'messages': messages}))
            rows = capture.for_case('r')
            assert rows['capture_errors'] == []
            assert rows['tool_results'][0]['result'] == json.loads(first[1]['content'])
            changed = exchange('c1', [evidence(text='Conflicting original')])
            await capture.on_request(httpx.Request('POST', 'https://model.local', json={'messages': changed}))
            assert capture.for_case('r')['capture_errors'] == ['ConflictingToolObservation']
    asyncio.run(scenario())


@pytest.mark.parametrize('size,budget,count', [(5000, 8000, 2), (18000, 64000, 4)])
def test_repeated_large_evidence_fits_budget_but_unique_content_still_stops(size, budget, count):
    from app.agent.loop_runner import AgentLoopRunner
    from app.agent.loop_models import LoopLimits
    from app.agent.loop_errors import LoopContextLimitError
    from app.observability.store import InMemoryTraceStore
    from app.tools.async_registry import AsyncTool, AsyncToolRegistry
    from app.agent.paper_tool import RetrievalInput
    from tests.test_agent_loop import turn

    async def scenario(unique):
        registry = AsyncToolRegistry()
        executions = []
        async def retrieve(args):
            executions.append(args.query)
            return {'evidence': [evidence(text=(str(len(executions)) if unique else 'x')*size)]}
        registry.register(AsyncTool('paper_retrieval', 'Retrieve', RetrievalInput, retrieve, read_only=True))
        histories = []
        turns = [turn((f'c{i}', 'paper_retrieval', {'query': str(i)})) for i in range(1, count+1)]
        turns.append(turn(content='Done'))
        async def model(messages, tools):
            histories.append(messages)
            return turns[len(histories)-1]
        runner = AgentLoopRunner(registry=registry, model=model, trace_store=InMemoryTraceStore(),
            limits=LoopLimits(max_history_chars=budget), evidence_history_tools=('paper_retrieval',))
        if unique:
            with pytest.raises(LoopContextLimitError):
                await runner.run('q', request_id='r', trace_id='t')
            assert len(histories) == count
        else:
            assert (await runner.run('q', request_id='r', trace_id='t')).answer == 'Done'
            assert len(histories) == count+1
            from app.agent.evidence_history import restore_evidence_history
            assert len(json.dumps(restore_evidence_history(histories[-1]), ensure_ascii=False)) > budget
            assert len(json.dumps(histories[-1], ensure_ascii=False)) <= budget
        assert executions == [str(i) for i in range(1, count+1)]
    asyncio.run(scenario(False))
    asyncio.run(scenario(True))
