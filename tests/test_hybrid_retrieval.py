import asyncio

import pytest

from app.config import Settings
from tests.answer_helpers import hit
from tests.test_rag_retrieve import make_rag


def test_explicit_strategy_is_validated_and_default_stays_vector():
    assert Settings(_env_file=None).rag_retrieval_strategy=='vector'
    assert Settings(_env_file=None,rag_retrieval_strategy='hybrid').rag_retrieval_strategy=='hybrid'
    with pytest.raises(ValueError):Settings(_env_file=None,rag_retrieval_strategy='typo')


def test_service_explicit_hybrid_dispatch_preserves_query_and_trace(tmp_path):
    rag,calls,traces=make_rag(tmp_path)
    rag._settings=Settings(_env_file=None,rag_storage_dir=tmp_path,rag_retrieval_strategy='hybrid')
    async def hybrid(paper_id,vector,spec,*,query,top_k):
        assert paper_id=='paper-1' and query=='datasets?' and top_k==3
        calls.append('hybrid')
        return ['hybrid hit']
    rag._store.search_hybrid=hybrid
    result=asyncio.run(rag.retrieve('paper-1','datasets?',top_k=3,request_id='r',trace_id='t'))
    assert result==['hybrid hit'] and calls[-1]=='hybrid' and 'search' not in calls
    trace=asyncio.run(traces.get('t'))
    assert trace.spans[-1].attributes['ranking_strategy']=='hybrid_rrf_v1'


def test_rrf_promotes_agreement_and_preserves_original_cosine_scores():
    from app.rag.hybrid import fuse_rankings
    first=hit(0);first.score=0.9
    second=hit(1,text='Keyword source.');second.score=0.1
    result=fuse_rankings([first,second],[second],top_k=2)
    assert result==[second,first] and result[0].score==0.1
    assert len({(h.chunk.page_number,h.chunk.chunk_index) for h in result})==2


def test_service_hybrid_failure_is_not_hidden_by_vector_fallback(tmp_path):
    rag,calls,traces=make_rag(tmp_path)
    rag._settings=Settings(_env_file=None,rag_storage_dir=tmp_path,rag_retrieval_strategy='hybrid')
    async def broken(*args,**kwargs):raise RuntimeError('lexical database unavailable')
    rag._store.search_hybrid=broken
    with pytest.raises(RuntimeError):
        asyncio.run(rag.retrieve('paper-1','datasets?',top_k=3,request_id='r',trace_id='t'))
    assert 'search' not in calls and asyncio.run(traces.get('t')).status=='error'


def test_service_ask_uses_same_selected_strategy(tmp_path):
    from types import SimpleNamespace
    rag,calls,traces=make_rag(tmp_path)
    rag._settings=Settings(_env_file=None,rag_storage_dir=tmp_path,rag_retrieval_strategy='hybrid')
    async def hybrid(*args,query,top_k):
        assert query=='datasets?' and top_k==3
        return ['hybrid evidence']
    expected=SimpleNamespace(citations=(),insufficient_evidence=True,generation_model=None)
    async def answer(paper_id,question,hits):
        assert paper_id=='paper-1' and question=='datasets?' and hits==['hybrid evidence']
        return expected
    rag._store.search_hybrid=hybrid
    rag._answer=SimpleNamespace(answer=answer)
    assert asyncio.run(rag.ask('paper-1','datasets?',top_k=3,request_id='r',trace_id='t')) is expected


def test_fusion_empty_lexical_order_ties_and_source_conflicts():
    from app.rag.hybrid import fuse_rankings
    one,two=hit(0),hit(1)
    assert fuse_rankings([one,two],[],top_k=1)==[one]
    assert fuse_rankings([one,two],[two,one],top_k=2)==[one,two]
    with pytest.raises(ValueError):fuse_rankings([one],[hit(0,text='Changed source')],top_k=1)
    with pytest.raises(ValueError):fuse_rankings([one],[hit(0,paper_id='foreign')],top_k=1)


def test_query_terms_are_bounded_safe_and_unsupported_text_can_fall_back():
    from app.rag.hybrid import lexical_query
    assert lexical_query('Datasets datasets NQ; DROP TABLE?')=='datasets OR nq OR drop OR table'
    assert lexical_query('这篇论文用了哪些数据集？')==''
    assert len(lexical_query(' '.join('term'+str(i) for i in range(100))).split(' OR '))==64
    for query in ('', ' ', 'x'*4001, None):
        with pytest.raises(ValueError):lexical_query(query)


@pytest.mark.parametrize('top_k',[0,101,True,1.5])
def test_hybrid_rejects_bad_limits_before_database(top_k):
    from app.rag.store import PostgresPaperStore
    from tests.test_paper_store import NoDatabase,sample
    info,_,vectors=sample()
    with pytest.raises(ValueError):
        asyncio.run(PostgresPaperStore(NoDatabase()).search_hybrid('paper',vectors[0],info.embedding,
            query='datasets',top_k=top_k))
