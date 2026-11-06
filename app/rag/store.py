"""Immutable paper imports and exact, single-paper cosine search."""
import json
import math
import struct

import asyncpg

from app.rag.cleaner import clean_page
from app.rag.models import PdfPage, PreparedText, TextChunk
from app.rag.splitter import split_page
from app.rag.storage_models import EmbeddingSpec, PaperInfo, RetrievedChunk, StoredPage, StoredPaper
from app.rag.threads import run_cpu
from app.rag.hybrid import fuse_rankings, lexical_query


class PaperAlreadyExistsError(Exception):
    pass


class PaperNotFoundError(Exception):
    pass


class EmbeddingMismatchError(Exception):
    pass


def _vector_text(value: list[float]) -> str:
    if not isinstance(value, list) or len(value) != 1024:
        raise ValueError("This schema requires a 1024-dimensional vector")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in value):
        raise ValueError("Vector values must be numeric")
    try:
        # pgvector stores float32. Reject overflow and zero-after-underflow too.
        values = [struct.unpack("!f", struct.pack("!f", float(x)))[0] for x in value]
    except (OverflowError, struct.error) as exc:
        raise ValueError("Vector values must fit float32") from exc
    if not all(math.isfinite(x) for x in values) or not any(values):
        raise ValueError("Vector must be finite and nonzero in float32")
    return json.dumps(values, separators=(",", ":"), allow_nan=False)


def _snapshot_import(info: PaperInfo, prepared: PreparedText, vectors: list[list[float]]):
    info = PaperInfo.model_validate(info.model_dump())
    prepared = PreparedText.model_validate(prepared.model_dump())
    if not prepared.pages or not prepared.chunks or len(vectors) != len(prepared.chunks):
        raise ValueError("A complete import requires pages, chunks and matching vectors")
    if info.cleaner_version != "conservative-v1" or info.splitter_version != "page-boundaries-v1":
        raise ValueError("Unsupported processing version; cannot verify import provenance")
    previous_page = 0
    expected_chunks = []
    for page in prepared.pages:
        if page.page_number <= previous_page:
            raise ValueError("Pages must be positive and strictly increasing")
        previous_page = page.page_number
        # Reproduce the declared pure-text versions before opening a DB
        # transaction. Internal slice consistency alone cannot prove provenance.
        canonical = clean_page(PdfPage(page_number=page.page_number, text=page.raw_text))
        if page != canonical:
            raise ValueError("Normalized page must be produced from its original text")
        expected_chunks.extend(split_page(canonical, chunk_size_chars=info.chunk_size_chars,
                                          overlap_chars=info.overlap_chars))
    if prepared.chunks != expected_chunks:
        raise ValueError("Chunks must reproduce the declared splitter configuration")
    vector_strings = [_vector_text(vector) for vector in vectors]
    return info, prepared, vector_strings


def _info(row) -> PaperInfo:
    return PaperInfo(
        paper_id=row["paper_id"], title=row["title"], source_path=row["source_path"],
        source_sha256=row["source_sha256"],
        embedding=EmbeddingSpec(model=row["embedding_model"], revision=row["embedding_revision"],
                                dimensions=row["embedding_dimensions"], query_instruction=row["query_instruction"],
                                representation_version=row["representation_version"]),
        cleaner_version=row["cleaner_version"], splitter_version=row["splitter_version"],
        chunk_size_chars=row["chunk_size_chars"], overlap_chars=row["overlap_chars"],
    )


def _chunk(row, raw_page: str) -> TextChunk:
    return TextChunk(
        page_number=row["page_number"], chunk_index=row["chunk_index"], text=row["text"],
        raw_text=raw_page[row["raw_start_char"]:row["raw_end_char"]],
        start_char=row["start_char"], end_char=row["end_char"],
        raw_start_char=row["raw_start_char"], raw_end_char=row["raw_end_char"],
    )


class PostgresPaperStore:
    """Borrow a pool managed in the caller's event loop; never close it here."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def search_hybrid(self, paper_id: str, query_vector: list[float], embedding: EmbeddingSpec,
                            *, query: str, top_k: int=5) -> list[RetrievedChunk]:
        """Rank two bounded candidate lists with RRF; hit.score remains cosine.

        No FTS schema/index is needed for the current bounded single-paper corpus.
        The legacy search method below retains its exact cosine ordering.
        """
        if type(top_k) is not int or not 1<=top_k<=100:
            raise ValueError('top_k must be an integer between 1 and 100')
        lexical=lexical_query(query)
        vector=_vector_text(query_vector)
        embedding=EmbeddingSpec.model_validate(embedding.model_dump())
        candidate_limit=max(25,top_k)
        async with self._pool.acquire() as connection:
            async with connection.transaction(isolation='repeatable_read',readonly=True):
                row=await connection.fetchrow('SELECT * FROM rag_papers WHERE paper_id=$1',paper_id)
                if row is None:raise PaperNotFoundError('Paper was not found')
                if _info(row).embedding!=embedding:
                    raise EmbeddingMismatchError('Query embedding identity does not match the paper index')
                vectors=await connection.fetch('''
                    SELECT c.*,p.raw_text AS page_raw_text,1-(c.embedding <=> $2::text::vector) AS score
                    FROM rag_chunks c JOIN rag_pages p
                      ON p.paper_id=c.paper_id AND p.page_number=c.page_number
                    WHERE c.paper_id=$1
                    ORDER BY c.embedding <=> $2::text::vector,c.page_number,c.chunk_index LIMIT $3
                ''',paper_id,vector,candidate_limit)
                words=[]
                if lexical:
                    words=await connection.fetch('''
                        WITH q AS (SELECT websearch_to_tsquery('english',$4) AS terms)
                        SELECT c.*,p.raw_text AS page_raw_text,1-(c.embedding <=> $2::text::vector) AS score
                        FROM rag_chunks c JOIN rag_pages p
                          ON p.paper_id=c.paper_id AND p.page_number=c.page_number CROSS JOIN q
                        WHERE c.paper_id=$1 AND to_tsvector('english',c.text) @@ q.terms
                        ORDER BY ts_rank_cd(to_tsvector('english',c.text),q.terms,2) DESC,
                                 c.page_number,c.chunk_index LIMIT $3
                    ''',paper_id,vector,candidate_limit,lexical)
                def restore(rows):
                    return [RetrievedChunk(paper_id=paper_id,chunk=_chunk(r,r['page_raw_text'].decode('utf-8')),
                                           score=r['score']) for r in rows]
                return fuse_rankings(restore(vectors),restore(words),top_k=top_k)

    async def save(self, info: PaperInfo, prepared: PreparedText, vectors: list[list[float]]) -> None:
        # Capture mutable containers before the first await. Pages/profile and
        # all scalar fields are immutable; mutable chunks/vectors need copies.
        captured = PreparedText.model_construct(pages=prepared.pages.copy(),
                                                chunks=[chunk.model_copy() for chunk in prepared.chunks])
        vector_snapshot = [vector.copy() if isinstance(vector,list) else vector for vector in vectors]
        info, prepared, vector_strings = await run_cpu(_snapshot_import,info,captured,vector_snapshot)
        profile = info.embedding
        try:
            async with self._pool.acquire() as connection:
                async with connection.transaction():
                    await connection.execute("""
                        INSERT INTO rag_papers (
                          paper_id,title,source_path,source_sha256,embedding_model,embedding_revision,
                          embedding_dimensions,query_instruction,representation_version,cleaner_version,
                          splitter_version,chunk_size_chars,overlap_chars
                        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
                    """, info.paper_id, info.title, info.source_path, info.source_sha256, profile.model,
                        profile.revision, profile.dimensions, profile.query_instruction, profile.representation_version,
                        info.cleaner_version, info.splitter_version, info.chunk_size_chars, info.overlap_chars)
                    await connection.executemany("""
                        INSERT INTO rag_pages (paper_id,page_number,raw_text,text) VALUES ($1,$2,$3,$4)
                    """, [(info.paper_id, p.page_number, p.raw_text.encode("utf-8"), p.text) for p in prepared.pages])
                    await connection.executemany("""
                        INSERT INTO rag_chunks (paper_id,page_number,chunk_index,text,start_char,end_char,
                                                raw_start_char,raw_end_char,embedding)
                        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9::text::vector)
                    """, [(info.paper_id, c.page_number, c.chunk_index, c.text, c.start_char, c.end_char,
                           c.raw_start_char, c.raw_end_char, vector) for c, vector in zip(prepared.chunks, vector_strings)])
        except asyncpg.UniqueViolationError as exc:
            if exc.constraint_name == "rag_papers_pkey":
                raise PaperAlreadyExistsError("Paper ID already exists; imports are immutable") from exc
            raise

    async def get_info(self, paper_id: str) -> PaperInfo | None:
        async with self._pool.acquire() as connection:
            row = await connection.fetchrow("SELECT * FROM rag_papers WHERE paper_id=$1", paper_id)
            return None if row is None else _info(row)

    async def get(self, paper_id: str) -> StoredPaper | None:
        async with self._pool.acquire() as connection:
            async with connection.transaction(isolation="repeatable_read", readonly=True):
                row = await connection.fetchrow("SELECT * FROM rag_papers WHERE paper_id=$1", paper_id)
                if row is None:
                    return None
                pages = [StoredPage(page_number=p["page_number"], raw_text=p["raw_text"].decode("utf-8"), text=p["text"])
                         for p in await connection.fetch(
                             "SELECT * FROM rag_pages WHERE paper_id=$1 ORDER BY page_number", paper_id)]
                raw_pages = {p.page_number: p.raw_text for p in pages}
                chunks = [_chunk(c, raw_pages[c["page_number"]]) for c in await connection.fetch(
                    "SELECT * FROM rag_chunks WHERE paper_id=$1 ORDER BY page_number,chunk_index", paper_id)]
                return StoredPaper(paper=_info(row), pages=pages, chunks=chunks)

    async def search(
        self, paper_id: str, query_vector: list[float], embedding: EmbeddingSpec, *, top_k: int = 5,
    ) -> list[RetrievedChunk]:
        if type(top_k) is not int or not 1 <= top_k <= 100:
            raise ValueError("top_k must be an integer between 1 and 100")
        vector = _vector_text(query_vector)
        embedding = EmbeddingSpec.model_validate(embedding.model_dump())
        async with self._pool.acquire() as connection:
            async with connection.transaction(isolation="repeatable_read", readonly=True):
                row = await connection.fetchrow("SELECT * FROM rag_papers WHERE paper_id=$1", paper_id)
                if row is None:
                    raise PaperNotFoundError("Paper was not found")
                if _info(row).embedding != embedding:
                    raise EmbeddingMismatchError("Query embedding identity does not match the paper index")
                rows = await connection.fetch("""
                    SELECT c.*, p.raw_text AS page_raw_text,
                           1 - (c.embedding <=> $2::text::vector) AS score
                    FROM rag_chunks c
                    JOIN rag_pages p ON p.paper_id=c.paper_id AND p.page_number=c.page_number
                    WHERE c.paper_id=$1
                    ORDER BY c.embedding <=> $2::text::vector, c.page_number, c.chunk_index
                    LIMIT $3
                """, paper_id, vector, top_k)
                return [RetrievedChunk(paper_id=paper_id, chunk=_chunk(r, r["page_raw_text"].decode("utf-8")),
                                       score=r["score"]) for r in rows]
