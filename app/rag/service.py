"""Synchronous-request RAG orchestration with bounded concurrent operations."""
import asyncio
from contextlib import asynccontextmanager
import hashlib
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

import pymupdf

from app.config import Settings
from app.rag.pdf import PdfReadError, extract_pdf_pages
from app.rag.preparation import prepare_pages
from app.rag.service_errors import RagBusyError, RagInputError, RagLimitError
from app.rag.service_models import PaperImportResult
from app.rag.storage_models import EmbeddingSpec, PaperInfo
from app.rag.store import EmbeddingMismatchError, PaperNotFoundError
from app.rag.tracing import RagRun, logger
from app.rag.threads import run_cpu as _thread_work


def _copy_upload(source: BinaryIO, destination: Path, max_bytes: int, state: dict):
    digest = hashlib.sha256()
    size = 0
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open("xb") as target:
        state["created"] = True
        while content := source.read(65536):
            size += len(content)
            if size > max_bytes:
                raise RagLimitError("PDF exceeds the file size limit")
            target.write(content)
            digest.update(content)
    if size == 0:
        raise RagInputError("Uploaded PDF is empty")
    return digest.hexdigest(), size


def _extract(path: Path, max_pages: int):
    try:
        with pymupdf.open(stream=path.read_bytes(),filetype="pdf") as doc:
            if doc.page_count > max_pages:
                raise RagLimitError("PDF exceeds the page limit")
    except pymupdf.FileDataError as exc:
        raise PdfReadError("Cannot read uploaded PDF") from exc
    return extract_pdf_pages(path)


def _prepare(pages, settings):
    if sum(len(p.text) for p in pages) > settings.rag_max_text_chars:
        raise RagLimitError("PDF exceeds the extracted text limit")
    result = prepare_pages(pages, chunk_size_chars=settings.rag_chunk_size_chars,
                           overlap_chars=settings.rag_overlap_chars)
    if len(result.chunks) > settings.rag_max_chunks:
        raise RagLimitError("PDF exceeds the chunk limit")
    return result


class RagService:
    def __init__(self, *, store, embedding, answer, trace_store, settings: Settings):
        self._store, self._embedding, self._answer = store, embedding, answer
        self._trace_store, self._settings = trace_store, settings
        self._storage = Path(settings.rag_storage_dir).resolve()
        self._capacity = asyncio.Semaphore(settings.rag_max_concurrency)

    @asynccontextmanager
    async def _admit(self, run):
        with run.stage("rag.admission"):
            try:
                async with asyncio.timeout(self._settings.rag_queue_timeout_seconds):
                    await self._capacity.acquire()
            except TimeoutError as exc:
                raise RagBusyError("RAG capacity is busy") from exc
        try:
            yield
        finally:
            self._capacity.release()

    async def _profile(self):
        return EmbeddingSpec(model=self._embedding.model,revision=await self._embedding.get_revision(),
                             dimensions=self._embedding.dimensions,query_instruction=self._embedding.query_instruction)

    async def import_pdf(self, source: BinaryIO, filename: str, title: str | None, *, request_id: str, trace_id: str):
        async with RagRun(self._trace_store,"rag.import",request_id,trace_id) as run:
            async with self._admit(run):
                if title is not None and (not isinstance(title,str) or not title.strip() or len(title)>300):
                    raise RagInputError("Title must be nonblank and at most 300 characters")
                display = (filename or "paper.pdf").replace("\\","/").rsplit("/",1)[-1]
                paper_id = str(uuid4())
                path = self._storage / (paper_id+".pdf")
                save_attempted = False
                file_state = {"created": False}
                try:
                    with run.stage("rag.upload") as attrs:
                        source_hash,size = await _thread_work(_copy_upload,source,path,self._settings.rag_max_file_bytes,file_state)
                        attrs["file_bytes"] = size
                    with run.stage("rag.pdf.extract") as attrs:
                        pages = await _thread_work(_extract,path,self._settings.rag_max_pages)
                        attrs["page_count"] = len(pages)
                    with run.stage("rag.text.prepare") as attrs:
                        prepared = await _thread_work(_prepare,pages,self._settings)
                        attrs["chunk_count"] = len(prepared.chunks)
                    with run.stage("rag.documents.embed",model=self._embedding.model) as attrs:
                        profile = await self._profile()
                        vectors = await self._embedding.embed_documents([c.text for c in prepared.chunks])
                        if await self._embedding.get_revision() != profile.revision:
                            raise EmbeddingMismatchError("Embedding model changed during import")
                        attrs["vector_count"] = len(vectors)
                    info = PaperInfo(paper_id=paper_id,title=title.strip() if title else (Path(display).stem or "Paper"),
                        source_path=str(path),source_sha256=source_hash,embedding=profile,
                        chunk_size_chars=self._settings.rag_chunk_size_chars,overlap_chars=self._settings.rag_overlap_chars)
                    with run.stage("rag.paper.save"):
                        save_attempted = True
                        await self._store.save(info,prepared,vectors)
                    run.attributes["paper_id"] = paper_id
                    return PaperImportResult(paper_id=paper_id,title=info.title,page_count=len(prepared.pages),
                                              chunk_count=len(prepared.chunks))
                except BaseException:
                    if save_attempted:
                        # The connection may fail after COMMIT. Keep the source
                        # rather than risk deleting a committed paper's PDF.
                        run.attributes["source_file_retained"] = True
                    elif file_state["created"]:
                        try:
                            await _thread_work(path.unlink,True)
                        except OSError as error:
                            logger.warning("rag.file_cleanup_failed error_type=%s",type(error).__name__)
                    raise

    async def ask(self, paper_id: str, question: str, *, top_k: int=5, request_id: str, trace_id: str):
        async with RagRun(self._trace_store,"rag.ask",request_id,trace_id) as run:
            async with self._admit(run):
                if not isinstance(question,str) or not question.strip() or len(question)>4000:
                    raise RagInputError("Question must be nonblank and at most 4000 characters")
                if type(top_k) is not int or not 1<=top_k<=100:
                    raise RagInputError("top_k must be an integer from 1 to 100")
                with run.stage("rag.paper.load"):
                    info = await self._store.get_info(paper_id)
                    if info is None:
                        raise PaperNotFoundError("Paper was not found")
                with run.stage("rag.question.embed",model=self._embedding.model):
                    profile = await self._profile()
                    if profile != info.embedding:
                        raise EmbeddingMismatchError("Current model identity differs from the imported index")
                    vector = await self._embedding.embed_query(question)
                    if await self._embedding.get_revision() != profile.revision:
                        raise EmbeddingMismatchError("Embedding model changed during question embedding")
                with run.stage("rag.paper.search",top_k=top_k) as attrs:
                    if self._settings.rag_retrieval_strategy=='hybrid':
                        hits=await self._store.search_hybrid(paper_id,vector,profile,query=question,top_k=top_k)
                        attrs['ranking_strategy']='hybrid_rrf_v1'
                    else:
                        hits=await self._store.search(paper_id,vector,profile,top_k=top_k)
                    attrs["hit_count"] = len(hits)
                with run.stage("rag.answer.generate") as attrs:
                    result = await self._answer.answer(paper_id,question,hits)
                    attrs["citation_count"] = len(result.citations)
                    attrs["insufficient_evidence"] = result.insufficient_evidence
                    if result.generation_model:
                        attrs["model"] = result.generation_model
                return result

    async def retrieve(self, paper_id: str, question: str, *, top_k: int=5, request_id: str, trace_id: str):
        """Expose persisted-index retrieval without invoking answer generation."""
        async with RagRun(self._trace_store,"rag.retrieve",request_id,trace_id) as run:
            async with self._admit(run):
                if not isinstance(question,str) or not question.strip() or len(question)>4000:
                    raise RagInputError("Question must be nonblank and at most 4000 characters")
                if type(top_k) is not int or not 1<=top_k<=100:
                    raise RagInputError("top_k must be an integer from 1 to 100")
                with run.stage("rag.paper.load"):
                    info=await self._store.get_info(paper_id)
                    if info is None:
                        raise PaperNotFoundError("Paper was not found")
                with run.stage("rag.question.embed",model=self._embedding.model):
                    profile=await self._profile()
                    if profile!=info.embedding:
                        raise EmbeddingMismatchError("Current model identity differs from the imported index")
                    vector=await self._embedding.embed_query(question)
                    if await self._embedding.get_revision()!=profile.revision:
                        raise EmbeddingMismatchError("Embedding model changed during question embedding")
                with run.stage("rag.paper.search",top_k=top_k) as attrs:
                    if self._settings.rag_retrieval_strategy=='hybrid':
                        hits=await self._store.search_hybrid(paper_id,vector,profile,query=question,top_k=top_k)
                        attrs['ranking_strategy']='hybrid_rrf_v1'
                    else:
                        hits=await self._store.search(paper_id,vector,profile,top_k=top_k)
                    attrs["hit_count"]=len(hits)
                return hits
