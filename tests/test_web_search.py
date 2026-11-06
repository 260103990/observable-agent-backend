import asyncio
import copy
import json

import httpx
import pytest
from pydantic import SecretStr

from app.observability.store import InMemoryTraceStore


def result(**changes):
    value={'title':'Cache study','url':'https://example.org/cache','snippet':'Short search snippet.',
           'summary':'A longer provider-generated search summary.','content':'','site_name':'Example',
           'site_icon':None,'date':'2026-11-01T00:00:00Z','rerank_score':2.3,'authority':3}
    value.update(changes)
    return value


def execute(response, **options):
    from app.tools.web_search import DeepSeekSearchClient
    requests=[]
    traces=InMemoryTraceStore()
    def handler(request):
        requests.append(request)
        if isinstance(response,BaseException): raise response
        return response
    async def run(query='cache site:usenix.org', count=5):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client=DeepSeekSearchClient(client=http,api_key=options.pop('api_key',SecretStr('unit-secret')),
                                       trace_store=traces,**options)
            batch=await client.search(query,count=count,request_id='request',trace_id='trace')
            assert not http.is_closed
            return batch
    return run,requests,traces


def test_search_contract_provenance_and_trace_body_bytes():
    response=httpx.Response(200,json={'results':[result()]})
    run,requests,traces=execute(response,timeout_seconds=7)
    batch=asyncio.run(run())
    request=requests[0]
    assert str(request.url)=='https://api.deepseek.com/v1/tools/search'
    assert json.loads(request.content)=={'query':'cache site:usenix.org','count':5,'need_content':False,'localization':''}
    assert request.headers['authorization']=='Bearer unit-secret'
    assert request.extensions['timeout']['read']==7
    assert batch.results[0].summary==result()['summary'] and batch.results[0].rank==1
    assert batch.results[0].source_type=='web_search_result'
    assert batch.results[0].published_at=='2026-11-01T00:00:00Z'
    assert batch.retrieved_at and batch.query=='cache site:usenix.org'
    trace=asyncio.run(traces.get('trace'))
    assert trace.spans[0].name=='web.search' and trace.status=='ok'
    assert trace.spans[0].attributes['response_body_bytes']==len(response.content)
    assert trace.spans[0].attributes['outbound_request_attempts']==1
    assert trace.spans[0].attributes['result_count']==1
    assert 'unit-secret' not in trace.model_dump_json() and 'cache site:usenix.org' not in trace.model_dump_json()


def test_empty_search_is_successful_and_counted():
    run,_,traces=execute(httpx.Response(200,json={'results':[]}))
    assert asyncio.run(run()).results==()
    assert asyncio.run(traces.get('trace')).spans[0].attributes['result_count']==0


@pytest.mark.parametrize('query,count',[
    ('',5),('  ',5),('x'*129,5),('q',0),('q',11),('q',True),('q',1.2),(None,5),
])
def test_invalid_input_rejected_before_http(query,count):
    run,requests,_=execute(httpx.Response(200,json={'results':[]}))
    with pytest.raises(ValueError): asyncio.run(run(query,count))
    assert requests==[]


@pytest.mark.parametrize('changes',[
    {'url':'javascript:alert(1)'},{'url':'file:///local'},{'url':'https://user:pass@example.org/a'},
    {'url':'https://example.org/has space'},{'url':'https://example.org:bad/a'},
    {'summary':None},{'title':None},{'snippet':None},{'summary':123},
    {'rerank_score':float('inf')},{'rerank_score':True},{'authority':True},
    {'date':'yesterday'},{'date':'2026-11-01T00:00:00'},{'content':'Unexpected full page body'},
])
def test_invalid_provider_results_rejected(changes):
    from app.tools.web_search import SearchResponseError
    body={'results':[result(**changes)]}
    # JSON cannot encode infinity with HTTPX; use a raw body to test parser rejection.
    response=httpx.Response(200,content=json.dumps(body).encode())
    run,_,traces=execute(response)
    with pytest.raises(SearchResponseError): asyncio.run(run())
    assert asyncio.run(traces.get('trace')).status=='error'


@pytest.mark.parametrize('body',[{}, {'results':None}, {'results':{}}, {'results':[{}]}, {'results':[result()]*6}])
def test_missing_or_excess_result_envelope_rejected(body):
    from app.tools.web_search import SearchResponseError
    run,_,_=execute(httpx.Response(200,json=body))
    with pytest.raises(SearchResponseError): asyncio.run(run())


@pytest.mark.parametrize('status',[302,400,401,402,403,408,429,500,504])
def test_provider_error_no_retry_no_redirect_no_body_leak(status):
    from app.tools.web_search import SearchResponseError
    run,requests,traces=execute(httpx.Response(status,text='unit-secret private response',headers={'location':'https://elsewhere.org'}))
    with pytest.raises(SearchResponseError) as caught: asyncio.run(run())
    assert caught.value.status_code==status and len(requests)==1
    assert 'private' not in str(caught.value) and 'unit-secret' not in str(caught.value)
    trace=asyncio.run(traces.get('trace'))
    assert trace.status=='error' and trace.spans[1].attributes['http_status']==status


@pytest.mark.parametrize('error,expected',[(httpx.ReadTimeout('private'),'SearchTimeoutError'),
                                        (httpx.ConnectError('private'),'SearchConnectionError')])
def test_transport_failure_is_distinct_safe_and_traced(error,expected):
    run,requests,traces=execute(error)
    with pytest.raises(Exception) as caught: asyncio.run(run())
    assert type(caught.value).__name__==expected and caught.value.__cause__ is error
    assert len(requests)==1 and asyncio.run(traces.get('trace')).status=='error'


def test_missing_key_sends_no_request():
    from app.tools.web_search import SearchConfigurationError
    run,requests,_=execute(None,api_key=None)
    with pytest.raises(SearchConfigurationError): asyncio.run(run())
    assert requests==[]


def test_response_stream_budget_stops_early_and_closes():
    from app.tools.web_search import SearchResponseError
    consumed,closed=[],[]
    class Large(httpx.AsyncByteStream):
        async def __aiter__(self):
            for index in range(20):
                consumed.append(index)
                yield b'x'*65536
        async def aclose(self): closed.append(True)
    run,_,traces=execute(httpx.Response(200,stream=Large()),max_response_bytes=131072)
    with pytest.raises(SearchResponseError): asyncio.run(run())
    assert len(consumed)==3 and closed==[True]
    assert asyncio.run(traces.get('trace')).spans[0].attributes['response_body_bytes']==196608


def test_external_cancellation_propagates_and_records_cancelled():
    from app.tools.web_search import DeepSeekSearchClient
    entered=asyncio.Event()
    traces=InMemoryTraceStore()
    async def run():
        async def handler(request):
            entered.set()
            await asyncio.Event().wait()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client=DeepSeekSearchClient(client=http,api_key=SecretStr('key'),trace_store=traces)
            task=asyncio.create_task(client.search('q',count=1,request_id='r',trace_id='t'))
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError): await task
            assert (await traces.get('t')).status=='cancelled'
    asyncio.run(run())


def test_trace_failure_preserves_search_success():
    from app.tools.web_search import DeepSeekSearchClient
    class Broken:
        async def save(self,trace): raise RuntimeError('private trace error')
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r:httpx.Response(200,json={'results':[]}))) as http:
            batch=await DeepSeekSearchClient(client=http,api_key=SecretStr('key'),trace_store=Broken()).search(
                'q',count=1,request_id='r',trace_id='t')
            assert batch.results==()
    asyncio.run(run())


def test_compressed_response_is_rejected_before_decompression_or_body_read():
    import gzip
    from app.tools.web_search import SearchResponseError
    consumed,closed=[],[]
    class Compressed(httpx.AsyncByteStream):
        async def __aiter__(self):
            consumed.append(True)
            yield gzip.compress(b'x'*(1024*1024))
        async def aclose(self): closed.append(True)
    run,requests,traces=execute(httpx.Response(200,headers={'content-encoding':'gzip'},stream=Compressed()))
    with pytest.raises(SearchResponseError): asyncio.run(run())
    assert consumed==[] and closed==[True]
    assert requests[0].headers['accept-encoding']=='identity'
    assert asyncio.run(traces.get('trace')).spans[0].attributes['response_body_bytes']==0
