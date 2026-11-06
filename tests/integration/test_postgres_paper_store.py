import asyncio
import os
from pathlib import Path
from uuid import uuid4

import asyncpg
import pytest

from tests.test_paper_store import sample


pytestmark = pytest.mark.postgres


async def open_pool():
    return await asyncpg.create_pool(dsn=os.environ["TEST_PGVECTOR_DATABASE_URL"], min_size=1, max_size=3)


async def initialize(pool):
    sql = Path(__file__).parents[2] / "app" / "rag" / "schema.sql"
    await pool.execute(sql.read_text(encoding="utf-8"))


async def clean(pool, ids):
    await pool.execute("DELETE FROM rag_papers WHERE paper_id = ANY($1::text[])", ids)


def test_save_restore_from_independent_pool_preserves_sources_and_blank_pages():
    from app.rag.store import PostgresPaperStore
    from app.rag.models import PdfPage
    from app.rag.preparation import prepare_pages

    info, _, _ = sample(f"paper-{uuid4()}")
    prepared = prepare_pages([PdfPage(page_number=1, text="  Cache\x00 hit.\r\n中文🙂"),
                              PdfPage(page_number=2, text=" \n"), PdfPage(page_number=3, text="Results: 42%.")],
                             chunk_size_chars=18, overlap_chars=2)
    vectors = [[1.0] + [0.0] * 1023 for _ in prepared.chunks]

    async def run():
        async with await open_pool() as pool:
            await initialize(pool)
            await PostgresPaperStore(pool).save(info, prepared, vectors)
        async with await open_pool() as pool:
            try:
                stored = await PostgresPaperStore(pool).get(info.paper_id)
                assert stored.paper == info
                assert [(p.page_number, p.raw_text, p.text) for p in stored.pages] == [
                    (p.page_number, p.raw_text, p.text) for p in prepared.pages]
                assert stored.chunks == prepared.chunks
                count = await pool.fetchval("SELECT count(*) FROM rag_chunks WHERE paper_id=$1", info.paper_id)
                assert count == len(vectors)
            finally:
                await clean(pool, [info.paper_id])
    asyncio.run(run())


def test_missing_paper_get_and_search_have_distinct_contracts():
    from app.rag.store import PaperNotFoundError, PostgresPaperStore

    info, _, vectors = sample()
    async def run():
        async with await open_pool() as pool:
            await initialize(pool)
            store = PostgresPaperStore(pool)
            missing = f"missing-{uuid4()}"
            assert await store.get(missing) is None
            with pytest.raises(PaperNotFoundError):
                await store.search(missing, vectors[0], info.embedding)
    asyncio.run(run())


def test_exact_cosine_search_filters_paper_orders_and_restores_original_text():
    from app.rag.store import PostgresPaperStore

    a, prepared, vectors = sample(f"paper-{uuid4()}")
    b, _, _ = sample(f"paper-{uuid4()}")
    assert len(vectors) >= 2
    vectors[0] = [0.0, 1.0] + [0.0] * 1022
    vectors[1] = [1.0, 0.0] + [0.0] * 1022
    async def run():
        async with await open_pool() as pool:
            await initialize(pool)
            store = PostgresPaperStore(pool)
            try:
                await store.save(a, prepared, vectors)
                await store.save(b, prepared, [[1.0] + [0.0] * 1023 for _ in vectors])
                hits = await store.search(a.paper_id, vectors[1], a.embedding, top_k=1)
                assert len(hits) == 1
                assert hits[0].paper_id == a.paper_id
                assert hits[0].chunk == prepared.chunks[1]
                assert hits[0].score == pytest.approx(1.0)
                all_hits = await store.search(a.paper_id, vectors[1], a.embedding, top_k=100)
                assert len(all_hits) == len(prepared.chunks)
                assert all(h.paper_id == a.paper_id for h in all_hits)
                assert [h.score for h in all_hits] == sorted((h.score for h in all_hits), reverse=True)
            finally:
                await clean(pool, [a.paper_id, b.paper_id])
    asyncio.run(run())


@pytest.mark.parametrize("field,value", [("model", "other-model"), ("revision", "other-revision"),
                                        ("query_instruction", "different instruction"),
                                        ("representation_version", "different-version")])
def test_search_rejects_embedding_identity_mismatch(field, value):
    from app.rag.store import EmbeddingMismatchError, PostgresPaperStore

    info, prepared, vectors = sample(f"paper-{uuid4()}")
    async def run():
        async with await open_pool() as pool:
            await initialize(pool)
            store = PostgresPaperStore(pool)
            try:
                await store.save(info, prepared, vectors)
                wrong = info.embedding.model_copy(update={field: value})
                with pytest.raises(EmbeddingMismatchError):
                    await store.search(info.paper_id, vectors[0], wrong)
            finally:
                await clean(pool, [info.paper_id])
    asyncio.run(run())


def test_duplicate_id_does_not_overwrite_and_concurrent_save_has_one_winner():
    from app.rag.store import PaperAlreadyExistsError, PostgresPaperStore

    info, prepared, vectors = sample(f"paper-{uuid4()}")
    async def run():
        async with await open_pool() as pool:
            await initialize(pool)
            store = PostgresPaperStore(pool)
            try:
                outcomes = await asyncio.gather(store.save(info, prepared, vectors),
                                                store.save(info, prepared, vectors), return_exceptions=True)
                assert sum(outcome is None for outcome in outcomes) == 1
                assert sum(isinstance(outcome, PaperAlreadyExistsError) for outcome in outcomes) == 1
                assert (await store.get(info.paper_id)).paper == info
                with pytest.raises(PaperAlreadyExistsError):
                    await store.save(info.model_copy(update={"title": "overwrite attempt"}), prepared, vectors)
                assert (await store.get(info.paper_id)).paper.title == info.title
            finally:
                await clean(pool, [info.paper_id])
    asyncio.run(run())


def test_late_database_failure_rolls_back_paper_pages_and_preceding_chunk():
    from app.rag.store import PostgresPaperStore

    info, prepared, vectors = sample(f"rollback-{uuid4()}")
    name = "reject_chunk_" + uuid4().hex
    async def run():
        async with await open_pool() as pool:
            await initialize(pool)
            # UUID-only generated identifiers/literal; never user SQL input.
            async with pool.acquire() as connection:
                await connection.execute(f"""
                    CREATE FUNCTION {name}() RETURNS trigger LANGUAGE plpgsql AS $$
                    BEGIN
                      IF NEW.paper_id = '{info.paper_id}' AND NEW.chunk_index = 1 THEN
                        RAISE EXCEPTION 'injected chunk failure';
                      END IF;
                      RETURN NEW;
                    END $$;
                    CREATE TRIGGER {name} BEFORE INSERT ON rag_chunks
                    FOR EACH ROW EXECUTE FUNCTION {name}();
                """)
            try:
                with pytest.raises(asyncpg.RaiseError, match="injected"):
                    await PostgresPaperStore(pool).save(info, prepared, vectors)
                for table in ("rag_papers", "rag_pages", "rag_chunks"):
                    assert await pool.fetchval(f"SELECT count(*) FROM {table} WHERE paper_id=$1", info.paper_id) == 0
            finally:
                await pool.execute(f"DROP TRIGGER {name} ON rag_chunks; DROP FUNCTION {name}();")
                await clean(pool, [info.paper_id])
    asyncio.run(run())
