import asyncio
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.observability.store import InMemoryTraceStore
from app.rag.service import RagService
from app.rag.storage_models import EmbeddingSpec


def make_rag(tmp_path, *, missing=False, mismatched=False, changing=False):
    calls=[]
    profile=EmbeddingSpec(model='embed',revision='r1',query_instruction='find')
    class Store:
        async def get_info(self,paper_id):
            calls.append('info')
            return None if missing else SimpleNamespace(embedding=profile)
        async def search(self,paper_id,vector,spec,*,top_k):
            calls.append('search')
            assert paper_id=='paper-1' and top_k==3 and spec==profile
            return ['a retrieved chunk']
    class Embed:
        model='embed'
        dimensions=1024
        query_instruction='find'
        async def get_revision(self):
            calls.append('revision')
            return 'r2' if mismatched or (changing and 'query' in calls) else 'r1'
        async def embed_query(self,question):
            calls.append('query')
            return [1.0]+[0.0]*1023
        async def embed_documents(self,*args):
            pytest.fail('Retrieval must not re-embed document chunks')
    class Answer:
        async def answer(self,*args):
            pytest.fail('Retrieval is not answer generation')
    traces=InMemoryTraceStore()
    rag=RagService(store=Store(),embedding=Embed(),answer=Answer(),trace_store=traces,
                   settings=Settings(_env_file=None,rag_storage_dir=tmp_path))
    return rag,calls,traces


def test_retrieve_reuses_persisted_index_without_generation(tmp_path):
    rag,calls,traces=make_rag(tmp_path)
    hits=asyncio.run(rag.retrieve('paper-1','question',top_k=3,request_id='r',trace_id='t'))
    assert hits==['a retrieved chunk'] and calls==['info','revision','query','revision','search']
    trace=asyncio.run(traces.get('t'))
    assert trace.status=='ok' and trace.spans[0].name=='rag.retrieve'
    assert len(trace.spans)==5


@pytest.mark.parametrize('option,expected',[
    ('missing','PaperNotFoundError'),('mismatched','EmbeddingMismatchError'),('changing','EmbeddingMismatchError'),
])
def test_retrieve_preserves_missing_and_index_identity_checks(tmp_path,option,expected):
    rag,calls,traces=make_rag(tmp_path,**{option:True})
    with pytest.raises(Exception) as caught:
        asyncio.run(rag.retrieve('paper-1','q',top_k=3,request_id='r',trace_id='t'))
    assert type(caught.value).__name__==expected and 'search' not in calls
    assert asyncio.run(traces.get('t')).status=='error'
