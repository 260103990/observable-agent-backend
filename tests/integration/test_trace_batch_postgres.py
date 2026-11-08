"""One batch statement, preserved Span order/types, and whole-trace rollback."""
import asyncio,os
from uuid import uuid4
import asyncpg
import pytest
from app.evaluation.trace_load import MeasuredPool,Sample,synthetic_trace
from app.observability.store import PostgresTraceStore

pytestmark=pytest.mark.postgres


@pytest.mark.parametrize('count',[0,1,4])
def test_span_batch_uses_one_insert_and_preserves_order_and_values(count):
    async def run():
        async with asyncpg.create_pool(os.environ['TEST_DATABASE_URL'],min_size=1,max_size=1) as pool:
            value=synthetic_trace('batch-'+uuid4().hex,count)
            for i,span in enumerate(value.spans):
                span.span_id='repeated-id';span.name='span-'+str(count-i)
                span.parent_span_id='external-parent' if i%2 else None
                span.error_type='example' if i%2 else None
                span.attributes={'text':'中文','flag':True,'number':i,'fraction':.5}
            sample=Sample(index=0);store=PostgresTraceStore(pool)
            try:
                await PostgresTraceStore(MeasuredPool(pool,sample,timeout=2)).save(value)
                assert sample.sql_calls==(3 if count else 2)
                assert await store.get(value.trace_id)==value
                await store.save(value.model_copy(update={'spans':[]}))
                assert (await store.get(value.trace_id)).spans==[]
            finally:await pool.execute('DELETE FROM traces WHERE trace_id=$1',value.trace_id)
    asyncio.run(run())


def test_invalid_batch_rolls_back_parent_overwrite_and_span_delete():
    async def run():
        async with asyncpg.create_pool(os.environ['TEST_DATABASE_URL'],min_size=1,max_size=1) as pool:
            store=PostgresTraceStore(pool);original=synthetic_trace('rollback-batch-'+uuid4().hex,2)
            await store.save(original)
            changed=original.model_copy(deep=True,update={'request_id':'replacement'})
            # Deliberately bypass model validation to inject a DB constraint failure.
            changed.spans[1]=changed.spans[1].model_copy(update={'name':None})
            try:
                with pytest.raises(asyncpg.NotNullViolationError):await store.save(changed)
                assert await store.get(original.trace_id)==original
            finally:await pool.execute('DELETE FROM traces WHERE trace_id=$1',original.trace_id)
    asyncio.run(run())
