import asyncio
import copy
import json

import httpx
import pytest
from pydantic import SecretStr


def envelope(*, calls=None, content=None, finish=None):
    return {'id':'provider-request','model':'deepseek-flash','choices':[{'finish_reason':finish or ('tool_calls' if calls else 'stop'),
        'message':{'role':'assistant','content':content,'tool_calls':calls}}]}


def call(identifier='c1', arguments='{"a":17,"b":25}'):
    return {'id':identifier,'type':'function','function':{'name':'add_numbers','arguments':arguments}}


def invoke(response, *, key=SecretStr('test-secret'), request_error=None):
    from app.agent.deepseek_loop import DeepSeekLoopModel
    requests=[]
    async def transport(request):
        requests.append(request)
        if request_error: raise request_error
        return response
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as http:
            model=DeepSeekLoopModel(client=http,api_key=key)
            return await model([{'role':'user','content':'Calculate'}],[])
    return run, requests


def test_native_tools_messages_and_authorization_are_preserved():
    from app.agent.deepseek_loop import DeepSeekLoopModel
    requests=[]
    async def run():
        def transport(request):
            requests.append(request)
            return httpx.Response(200,json=envelope(calls=[call()]))
        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as http:
            model=DeepSeekLoopModel(client=http,api_key=SecretStr('test-secret'),timeout_seconds=12)
            tools=[{'type':'function','function':{'name':'add_numbers','parameters':{'type':'object'}}}]
            messages=[{'role':'user','content':'Calculate'},{'role':'assistant','content':None,'tool_calls':[call('old')]},
                {'role':'tool','tool_call_id':'old','content':'{"ok":true,"output":42}'}]
            snapshot=copy.deepcopy(messages)
            turn=await model(messages,tools)
            assert messages==snapshot
            assert turn.calls[0].arguments=={'a':17,'b':25}
            body=json.loads(requests[0].content)
            assert body['messages']==messages and body['tools']==tools
            assert body['thinking']=={'type':'disabled'} and body['tool_choice']=='auto' and body['stream'] is False
            assert requests[0].headers['authorization']=='Bearer test-secret'
            assert requests[0].extensions['timeout']['read']==12
    asyncio.run(run())


def test_final_text_is_accepted():
    action,_=invoke(httpx.Response(200,json=envelope(content='42')))
    turn=asyncio.run(action())
    assert turn.content=='42' and turn.calls==()


@pytest.mark.parametrize('body',[
    {}, {'choices':[]}, envelope(content='42',finish='length'), envelope(content='42',finish='content_filter'),
    envelope(content=' '), envelope(calls=[call(),call()]),
    envelope(calls=[call(arguments='[]')]), envelope(calls=[call(arguments='{broken')]),
    envelope(calls=[call(arguments='{"a":NaN}')]), envelope(calls=[call(arguments='{"a":1,"a":2}')]),
    envelope(calls=[call(arguments='{"a":"'+'x'*9000+'"}')]),
    envelope(calls=[{'id':'x','type':'other','function':{'name':'add_numbers','arguments':'{}'}}]),
    envelope(calls=[call()],finish='stop'), envelope(content='42',finish='tool_calls'),
])
def test_invalid_or_truncated_provider_response_is_rejected(body):
    from app.agent.loop_errors import LoopProtocolError
    action,_=invoke(httpx.Response(200,json=body))
    with pytest.raises(LoopProtocolError): asyncio.run(action())


@pytest.mark.parametrize('status',[302,400,401,429,500])
def test_remote_failure_is_not_retried_redirected_or_echoed(status):
    from app.agent.loop_errors import LoopProtocolError
    action,requests=invoke(httpx.Response(status,text='test-secret private response',headers={'location':'https://other.example'}))
    with pytest.raises(LoopProtocolError) as caught: asyncio.run(action())
    assert len(requests)==1 and 'test-secret' not in str(caught.value) and 'private' not in str(caught.value)


@pytest.mark.parametrize('error,expected',[(httpx.ReadTimeout('private'),'LoopTimeoutError'),
                                        (httpx.ConnectError('private'),'LoopConnectionError')])
def test_transport_failure_has_safe_type_and_preserved_cause(error,expected):
    action,_=invoke(None,request_error=error)
    with pytest.raises(Exception) as caught: asyncio.run(action())
    assert type(caught.value).__name__==expected and caught.value.__cause__ is error
    assert 'private' not in str(caught.value)


def test_missing_key_fails_before_request():
    from app.agent.loop_errors import LoopConfigurationError
    action,requests=invoke(None,key=None)
    with pytest.raises(LoopConfigurationError): asyncio.run(action())
    assert requests==[]


def test_oversize_response_stops_stream_early_and_closes_it():
    from app.agent.deepseek_loop import DeepSeekLoopModel
    from app.agent.loop_errors import LoopProtocolError
    consumed=[]
    closed=[]
    class LargeBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            for index in range(32):
                consumed.append(index)
                yield b'x'*65536
        async def aclose(self):
            closed.append(True)
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r:httpx.Response(200,stream=LargeBody()))) as http:
            with pytest.raises(LoopProtocolError):
                await DeepSeekLoopModel(client=http,api_key=SecretStr('unit-key'))([],[])
    asyncio.run(run())
    assert len(consumed)<=3 and closed==[True]
