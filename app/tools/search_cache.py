"""Bounded process-local Search snapshots. No network clients, credentials or tasks are retained."""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math
from time import monotonic

from app.tools.web_models import SearchBatch, SearchInput
from app.tools.search_coalescing import SearchSingleFlight


CACHE_CONTRACT_VERSION=1
SEARCH_ENDPOINT='https://api.deepseek.com/v1/tools/search'


def search_cache_key(query,count,*,api_key,max_response_bytes):
    arguments=SearchInput(query=query,count=count)
    profile={'contract_version':CACHE_CONTRACT_VERSION,'provider':'deepseek','endpoint':SEARCH_ENDPOINT,
        'method':'POST','parameters':{'query':arguments.query,'count':arguments.count,
                                     'need_content':False,'localization':''},
        'credential_scope':hashlib.sha256(api_key.get_secret_value().encode('utf-8')).hexdigest(),
        'max_response_bytes':max_response_bytes}
    return hashlib.sha256(json.dumps(profile,sort_keys=True,ensure_ascii=False,
        separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class CachedSearch:
    batch: SearchBatch
    age_seconds: float


@dataclass(frozen=True)
class _Entry:
    payload: bytes
    stored_at: float
    expires_at: float


class MemorySearchCache:
    """Single-event-loop cache; methods do not yield or do I/O. Capacity measures serialized bytes, not RSS."""
    def __init__(self,*,ttl_seconds=300.0,max_entries=128,max_bytes=8*1024*1024,
                 max_entry_bytes=262144,clock=monotonic,coalesce_enabled=True,max_inflight=64,max_waiters=32):
        if (isinstance(ttl_seconds,bool) or not isinstance(ttl_seconds,(int,float))
                or not math.isfinite(ttl_seconds) or ttl_seconds<=0):
            raise ValueError('Cache TTL must be positive and finite')
        if any(type(value) is not int or value<=0 for value in (max_entries,max_bytes,max_entry_bytes)):
            raise ValueError('Cache capacity must contain positive integers')
        self.ttl_seconds=ttl_seconds
        self._max_entries,self._max_bytes,self._max_entry_bytes=max_entries,max_bytes,max_entry_bytes
        self._clock=clock
        self._entries=OrderedDict()
        self._bytes=0
        self.generation=0
        self.flights=SearchSingleFlight(enabled=coalesce_enabled,max_keys=max_inflight,max_waiters=max_waiters)

    def _remove(self,key):
        old=self._entries.pop(key,None)
        if old is not None:self._bytes-=len(old.payload)

    async def get(self,key):
        entry=self._entries.get(key)
        if entry is None:return None
        now=self._clock()
        if now>=entry.expires_at:
            self._remove(key)
            return None
        self._entries.move_to_end(key)
        return CachedSearch(SearchBatch.model_validate_json(entry.payload),max(0.0,now-entry.stored_at))

    async def put(self,key,batch):
        if not isinstance(batch,SearchBatch):raise TypeError('Cache requires a validated SearchBatch')
        payload=batch.model_dump_json().encode('utf-8')
        self._remove(key)
        if not batch.results or len(payload)>min(self._max_bytes,self._max_entry_bytes):return False
        now=self._clock()
        for old,entry in list(self._entries.items()):
            if now>=entry.expires_at:self._remove(old)
        while self._entries and (len(self._entries)>=self._max_entries or self._bytes+len(payload)>self._max_bytes):
            self._remove(next(iter(self._entries)))
        self._entries[key]=_Entry(payload,now,now+self.ttl_seconds)
        self._bytes+=len(payload)
        return True

    def clear(self):
        self.generation+=1
        self.flights.clear()
        self._entries.clear()
        self._bytes=0


def cache_from_settings(settings):
    return MemorySearchCache(ttl_seconds=settings.web_search_cache_ttl_seconds,
        max_entries=settings.web_search_cache_max_entries,max_bytes=settings.web_search_cache_max_bytes,
        max_entry_bytes=settings.web_search_cache_max_entry_bytes,
        coalesce_enabled=settings.web_search_cache_coalesce_enabled,
        max_inflight=settings.web_search_cache_max_inflight,max_waiters=settings.web_search_cache_max_waiters)
