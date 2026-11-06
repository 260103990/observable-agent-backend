import asyncio
import os
from uuid import uuid4

import asyncpg
import pytest

from app.rag.models import PdfPage
from app.rag.preparation import prepare_pages
from tests.test_paper_store import sample

pytestmark=pytest.mark.postgres


def test_hybrid_real_fts_fuses_lower_cosine_keyword_hit_and_keeps_paper_scope():
    from app.rag.store import PostgresPaperStore,PaperNotFoundError,EmbeddingMismatchError
    url=os.getenv('TEST_PGVECTOR_DATABASE_URL')
    if not url:pytest.skip('TEST_PGVECTOR_DATABASE_URL is not configured')
    assert url.rsplit('/',1)[-1].endswith('_test')
    info,_,_=sample('hybrid-'+uuid4().hex)
    info=info.model_copy(update={'chunk_size_chars':1200,'overlap_chars':150})
    other=info.model_copy(update={'paper_id':'foreign-'+uuid4().hex})
    prepared=prepare_pages([PdfPage(page_number=1,text='Common latency speed overview.'),
                            PdfPage(page_number=2,text='Quartzzephyr benchmark measured ten runs.')])
    vectors=[[1.0]+[0.0]*1023,[0.0,1.0]+[0.0]*1022]
    async def scenario():
        async with asyncpg.create_pool(url,min_size=1,max_size=2) as pool:
            store=PostgresPaperStore(pool)
            try:
                await store.save(info,prepared,vectors)
                await store.save(other,prepared,vectors)
                vector=await store.search(info.paper_id,vectors[0],info.embedding,top_k=1)
                hybrid=await store.search_hybrid(info.paper_id,vectors[0],info.embedding,query='Quartzzephyr',top_k=1)
                assert vector[0].chunk.page_number==1
                assert hybrid[0].chunk==prepared.chunks[1] and hybrid[0].paper_id==info.paper_id
                assert hybrid[0].score==pytest.approx(0.0)  # Still cosine, not fusion score.
                assert hybrid[0].chunk.raw_text=='Quartzzephyr benchmark measured ten runs.'
                assert await store.search_hybrid(info.paper_id,vectors[0],info.embedding,query='全文检索',top_k=1)==vector
                # User text is data; this must not execute SQL, even when it contains quotes/operators.
                await store.search_hybrid(info.paper_id,vectors[0],info.embedding,query="x'); DROP TABLE rag_papers; --",top_k=1)
                assert await store.get_info(other.paper_id) is not None
                with pytest.raises(PaperNotFoundError):
                    await store.search_hybrid('missing-'+uuid4().hex,vectors[0],info.embedding,query='q')
                wrong=info.embedding.model_copy(update={'revision':'wrong'})
                with pytest.raises(EmbeddingMismatchError):
                    await store.search_hybrid(info.paper_id,vectors[0],wrong,query='q')
            finally:
                await pool.execute('DELETE FROM rag_papers WHERE paper_id=ANY($1::text[])',[info.paper_id,other.paper_id])
    asyncio.run(scenario())
