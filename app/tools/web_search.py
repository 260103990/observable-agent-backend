"""Direct DeepSeek Search API, with independent provenance and request measurements."""
import asyncio
from datetime import datetime, timezone
import math

import httpx
from pydantic import SecretStr

from app.agent.loop_models import load_json_object
from app.agent.loop_trace import LoopTrace
from app.tools.web_models import SearchBatch, SearchInput, SearchResult
from app.tools.search_cache import CachedSearch, SEARCH_ENDPOINT, search_cache_key


SEARCH_URL=SEARCH_ENDPOINT


class SearchError(Exception):
    pass


class SearchConfigurationError(SearchError):
    pass


class SearchResponseError(SearchError):
    def __init__(self,message,*,status_code=None):
        super().__init__(message)
        self.status_code=status_code


class SearchTimeoutError(TimeoutError,SearchError):
    pass


class SearchConnectionError(SearchError):
    pass


class DeepSeekSearchClient:
    def __init__(self,*,client,api_key,trace_store,timeout_seconds=30.0,
                 max_response_bytes=262144,trace_timeout_seconds=5.0,cache=None,cache_timeout_seconds=0.05):
        if (isinstance(timeout_seconds,bool) or not isinstance(timeout_seconds,(int,float))
                or not math.isfinite(timeout_seconds) or timeout_seconds<=0):
            raise ValueError('Search timeout must be positive and finite')
        if type(max_response_bytes) is not int or max_response_bytes<=0:
            raise ValueError('Search response budget must be a positive integer')
        if api_key is not None and (not isinstance(api_key,SecretStr) or not api_key.get_secret_value().strip()):
            raise ValueError('API key must be a nonblank SecretStr or None')
        if (isinstance(cache_timeout_seconds,bool) or not isinstance(cache_timeout_seconds,(int,float))
                or not math.isfinite(cache_timeout_seconds) or cache_timeout_seconds<=0):
            raise ValueError('Cache operation deadline must be positive and finite')
        self._http,self._key,self._store=client,api_key,trace_store
        self._timeout,self._max_bytes,self._trace_timeout=timeout_seconds,max_response_bytes,trace_timeout_seconds
        self._cache,self._cache_timeout=cache,cache_timeout_seconds

    async def search(self,query,*,count=5,request_id,trace_id):
        arguments=SearchInput(query=query,count=count)
        if self._key is None:
            raise SearchConfigurationError('DeepSeek Search requires a configured API key')
        async with LoopTrace(self._store,request_id,trace_id,timeout_seconds=self._trace_timeout,
                             root_name='web.search') as trace:
            trace.attributes.update(provider='deepseek',outbound_request_attempts=0,response_body_bytes=0,
                                    cache_status='bypass' if self._cache is None else 'miss')
            generation=getattr(self._cache,'generation',None)
            cache_key=None
            if self._cache is not None:
                cache_key=search_cache_key(arguments.query,arguments.count,api_key=self._key,max_response_bytes=self._max_bytes)
                trace.attributes['cache_key_hash']=cache_key
                try:
                    with trace.stage('search.cache.get'):
                        async with asyncio.timeout(self._cache_timeout):
                            cached=await self._cache.get(cache_key)
                        if cached is not None:
                            if (not isinstance(cached,CachedSearch) or not isinstance(cached.batch,SearchBatch)
                                    or cached.batch.query!=arguments.query or not cached.batch.results
                                    or len(cached.batch.results)>arguments.count
                                    or cached.batch.response_body_bytes>self._max_bytes):
                                raise ValueError('Invalid cached Search snapshot')
                            trace.attributes.update(cache_status='hit',origin_search_trace_id=cached.batch.trace_id,
                                cache_age_seconds=cached.age_seconds,result_count=len(cached.batch.results))
                            return cached.batch.model_copy(update={'trace_id':trace_id,'response_body_bytes':0})
                except Exception as exc:
                    trace.attributes.update(cache_status='error',cache_error_type=type(exc).__name__)
            flights=getattr(self._cache,'flights',None)
            ticket=flights.claim(cache_key) if flights is not None and flights.enabled else None
            if ticket is None:
                trace.attributes['coalescing_status']='disabled' if flights is None or not flights.enabled else 'capacity_bypass'
            else:
                trace.attributes['coalescing_role']='leader' if ticket.leader else 'follower'
                trace.attributes['coalescing_status']='owner' if ticket.leader else 'waiting'
            if ticket is not None and not ticket.leader:
                try:
                    with trace.stage('search.coalescing.wait'):
                        try:
                            async with asyncio.timeout(self._timeout):
                                outcome=await asyncio.shield(ticket.future)
                        except TimeoutError as exc:
                            trace.attributes['coalescing_status']='wait_timeout'
                            raise SearchTimeoutError('Waiting for shared Search timed out') from exc
                        if outcome.error or outcome.batch is None:
                            trace.attributes['coalescing_status']=outcome.error or 'failed'
                            if outcome.error=='timeout':raise SearchTimeoutError('Shared Search timed out')
                            raise SearchConnectionError('Shared Search did not complete successfully')
                        shared=outcome.batch
                        if (shared.query!=arguments.query or len(shared.results)>arguments.count
                                or shared.response_body_bytes>self._max_bytes):
                            raise SearchResponseError('Shared Search returned an invalid snapshot')
                        trace.attributes.update(coalescing_status='reused',origin_search_trace_id=shared.trace_id,
                                                result_count=len(shared.results))
                        return shared.model_copy(update={'trace_id':trace_id,'response_body_bytes':0})
                finally:
                    flights.release_waiter(ticket)
            completed=None
            failure='failed'
            try:
                completed=await self._fetch(arguments,trace,trace_id,cache_key,generation)
                return completed
            except SearchTimeoutError:
                failure='timeout'
                raise
            finally:
                if ticket is not None:
                    flights.finish(ticket,batch=completed,error=None if completed is not None else failure)

    async def _fetch(self,arguments,trace,trace_id,cache_key,generation):
        with trace.stage('search.http',provider='deepseek') as attrs:
            try:
                async with asyncio.timeout(self._timeout):
                    trace.attributes['outbound_request_attempts']=1
                    async with self._http.stream('POST',SEARCH_URL,json={
                        'query':arguments.query,'count':arguments.count,'need_content':False,'localization':''},
                        headers={'Authorization':'Bearer '+self._key.get_secret_value(),'Accept-Encoding':'identity'},
                        timeout=self._timeout,follow_redirects=False) as response:
                        attrs['http_status']=response.status_code
                        response.raise_for_status()
                        if response.headers.get('content-encoding','identity').strip().lower()!='identity':
                            raise SearchResponseError('Compressed Search responses are not supported')
                        chunks=[]
                        async for chunk in response.aiter_bytes():
                            trace.attributes['response_body_bytes']+=len(chunk)
                            if trace.attributes['response_body_bytes']>self._max_bytes:
                                raise SearchResponseError('Search response exceeded its byte budget')
                            chunks.append(chunk)
                        raw=b''.join(chunks)
            except (httpx.TimeoutException,TimeoutError) as exc:
                raise SearchTimeoutError('DeepSeek Search timed out') from exc
            except httpx.RequestError as exc:
                raise SearchConnectionError('DeepSeek Search transport failed') from exc
            except httpx.HTTPStatusError as exc:
                raise SearchResponseError('DeepSeek Search returned an HTTP error',
                                          status_code=exc.response.status_code) from exc
        with trace.stage('search.validate'):
            try:
                body=load_json_object(raw.decode('utf-8'))
                records=body['results']
                if not isinstance(records,list) or len(records)>arguments.count:
                    raise ValueError('Invalid Search results count')
                results=[]
                for rank,record in enumerate(records,1):
                    if record['content']!='':
                        raise ValueError('Search unexpectedly returned fetched content')
                    results.append(SearchResult(rank=rank,title=record['title'],url=record['url'],
                        snippet=record['snippet'],summary=record['summary'],published_at=record.get('date'),
                        rerank_score=record['rerank_score'],authority=record['authority']))
            except (KeyError,ValueError,TypeError,OverflowError,RecursionError) as exc:
                raise SearchResponseError('DeepSeek Search returned an invalid result structure') from exc
            trace.attributes['result_count']=len(results)
            batch=SearchBatch(query=arguments.query,retrieved_at=datetime.now(timezone.utc).isoformat(),
                trace_id=trace_id,response_body_bytes=trace.attributes['response_body_bytes'],results=tuple(results))
        if self._cache is not None and batch.results:
            if generation!=getattr(self._cache,'generation',None):
                trace.attributes['cache_write_status']='invalidated'
                return batch
            try:
                with trace.stage('search.cache.put'):
                    async with asyncio.timeout(self._cache_timeout):
                        admitted=await self._cache.put(cache_key,batch)
                    trace.attributes['cache_write_status']='stored' if admitted else 'skipped'
            except Exception as exc:
                trace.attributes.update(cache_write_status='error',cache_error_type=type(exc).__name__)
        return batch
