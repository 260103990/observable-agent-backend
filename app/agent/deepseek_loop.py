"""Native Chat Completions tool calls; borrowed HTTP client, no hidden retries."""
import math

import httpx
from pydantic import SecretStr, ValidationError

from app.agent.loop_errors import LoopConfigurationError, LoopConnectionError, LoopProtocolError, LoopTimeoutError
from app.agent.loop_models import LoopToolCall, LoopTurn, load_json_object


class DeepSeekLoopModel:
    def __init__(self, *, client: httpx.AsyncClient, api_key: SecretStr | None,
                 model='deepseek-flash', chat_url='https://api.deepseek.com/chat/completions',
                 timeout_seconds=120.0, max_tokens=2048, max_calls_per_turn=4):
        if not isinstance(model,str) or not model.strip() or not isinstance(chat_url,str) or not chat_url.strip():
            raise ValueError('Model and chat URL must not be blank')
        if (isinstance(timeout_seconds,bool) or not isinstance(timeout_seconds,(int,float))
                or not math.isfinite(timeout_seconds) or timeout_seconds<=0):
            raise ValueError('Model timeout must be positive and finite')
        if any(type(n) is not int or n<=0 for n in (max_tokens,max_calls_per_turn)):
            raise ValueError('Model limits must be positive integers')
        if api_key is not None and (not isinstance(api_key,SecretStr) or not api_key.get_secret_value().strip()):
            raise ValueError('API key must be a nonblank SecretStr or None')
        self._http, self._key, self._model, self._url = client, api_key, model, chat_url
        self._timeout, self._tokens, self._max_calls = timeout_seconds, max_tokens, max_calls_per_turn

    async def __call__(self, messages, tools):
        if self._key is None:
            raise LoopConfigurationError('DEEPSEEK_API_KEY is required for Agent Loop')
        payload = {'model':self._model, 'messages':messages, 'tools':tools, 'tool_choice':'auto',
                   'thinking':{'type':'disabled'}, 'stream':False, 'max_tokens':self._tokens}
        try:
            async with self._http.stream('POST',self._url,json=payload,
                headers={'Authorization':'Bearer '+self._key.get_secret_value()},
                timeout=self._timeout,follow_redirects=False) as response:
                response.raise_for_status()
                chunks, size = [], 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size>131072:
                        raise LoopProtocolError('DeepSeek response exceeded its byte budget')
                    chunks.append(chunk)
                raw=b''.join(chunks)
        except httpx.TimeoutException as exc:
            raise LoopTimeoutError('DeepSeek loop request timed out') from exc
        except httpx.RequestError as exc:
            raise LoopConnectionError('DeepSeek loop transport failed') from exc
        except httpx.HTTPStatusError as exc:
            raise LoopProtocolError('DeepSeek loop returned an HTTP error') from exc
        try:
            body=load_json_object(raw.decode('utf-8'))
            if not isinstance(body,dict) or not isinstance(body.get('choices'),list) or len(body['choices'])!=1:
                raise ValueError('Expected one choice')
            if any(not isinstance(body.get(k),str) or not body[k].strip() for k in ('id','model')):
                raise ValueError('Missing provider metadata')
            choice=body['choices'][0]
            message=choice['message']
            if message['role']!='assistant':
                raise ValueError('Expected an assistant message')
            content=message.get('content')
            if content is not None and not isinstance(content,str):
                raise ValueError('Invalid assistant content')
            calls=message.get('tool_calls')
            if choice['finish_reason']=='stop' and not calls:
                return LoopTurn(content=content)
            if choice['finish_reason']!='tool_calls' or not isinstance(calls,list) or not 1<=len(calls)<=self._max_calls:
                raise ValueError('Expected complete bounded tool calls')
            result=[]
            for call in calls:
                if call['type']!='function':
                    raise ValueError('Unsupported tool type')
                arguments=call['function']['arguments']
                if not isinstance(arguments,str) or len(arguments)>8192:
                    raise ValueError('Invalid tool arguments')
                result.append(LoopToolCall(id=call['id'], name=call['function']['name'],
                                           arguments=load_json_object(arguments)))
            if len({call.id for call in result})!=len(result):
                raise ValueError('Duplicate call ID')
            return LoopTurn(content=content,calls=tuple(result))
        except (KeyError, TypeError, ValueError, ValidationError, OverflowError, RecursionError) as exc:
            raise LoopProtocolError('DeepSeek returned an invalid or incomplete loop decision') from exc
