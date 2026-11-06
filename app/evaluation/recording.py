"""Evaluation-only capture. Never export headers, credentials or all settings."""
import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from uuid import uuid4

from app.agent.loop_models import load_json_object
from app.agent.evidence_history import restore_evidence_history
from app.tools.web_search import SEARCH_URL


class EvaluationRecorder:
    def __init__(self):
        self._traces={}
        self._lock=asyncio.Lock()

    async def save(self,trace):
        async with self._lock:
            self._traces[trace.trace_id]=trace.model_copy(deep=True)

    async def get(self,trace_id):
        async with self._lock:
            trace=self._traces.get(trace_id)
            return trace.model_copy(deep=True) if trace is not None else None

    async def for_request(self,request_id):
        async with self._lock:
            return [trace.model_copy(deep=True) for trace in self._traces.values() if trace.request_id==request_id]


class EvaluationCapture:
    def __init__(self,*,model_url):
        self._model_url=model_url
        self._active=ContextVar('evaluation_request_id',default=None)
        self._cases={}
        self._search_index={}

    @contextmanager
    def for_request(self,request_id):
        token=self._active.set(request_id)
        self._cases.setdefault(request_id,{'search_requests':[],'tool_results':{},'model_request_count':0,
                                          'model_decisions':[],'capture_errors':[]})
        try:
            yield
        finally:
            self._active.reset(token)

    def for_case(self,request_id):
        value=deepcopy(self._cases.get(request_id,{'search_requests':[],'tool_results':{},'model_request_count':0,
                                                 'model_decisions':[],'capture_errors':[]}))
        value['tool_results']=list(value['tool_results'].values())
        return value

    def record_decision(self,decision):
        request_id=self._active.get()
        if request_id is not None:
            self._cases[request_id]['model_decisions'].append(decision.model_dump())

    async def on_request(self,request):
        request_id=self._active.get()
        if request_id is None:
            return
        value=self._cases[request_id]
        try:
            endpoint=str(request.url)
            if endpoint not in (SEARCH_URL,self._model_url):
                return
            body=load_json_object(request.content.decode('utf-8'))
            if endpoint==SEARCH_URL:
                entry={'endpoint':SEARCH_URL,'method':request.method,
                       'parameters':{k:body[k] for k in ('query','count','need_content','localization')},'http_status':None}
                identifier=str(uuid4())
                request.extensions['evaluation_search_id']=identifier
                self._search_index[identifier]=entry
                value['search_requests'].append(entry)
                return
            value['model_request_count']+=1
            calls={}
            for message in restore_evidence_history(body['messages']):
                if message['role']=='assistant':
                    for call in message.get('tool_calls') or []:
                        calls[call['id']]={'name':call['function']['name'],
                                          'arguments':load_json_object(call['function']['arguments'])}
                elif message['role']=='tool':
                    call_id=message['tool_call_id']
                    observation={'call_id':call_id,**calls[call_id],
                                 'result':load_json_object(message['content'])}
                    previous=value['tool_results'].get(call_id)
                    if previous is not None and previous!=observation:
                        value['capture_errors'].append('ConflictingToolObservation')
                    else:
                        value['tool_results'][call_id]=observation
        except (ValueError,KeyError,TypeError,UnicodeError,RecursionError) as exc:
            # Observation problems must be explicit without changing business execution.
            value['capture_errors'].append(type(exc).__name__)

    async def on_response(self,response):
        identifier=response.request.extensions.get('evaluation_search_id')
        if identifier in self._search_index:
            self._search_index[identifier]['http_status']=response.status_code
