import asyncio
from io import BytesIO
from pathlib import Path

import pytest

from app.config import Settings
from app.observability.store import InMemoryTraceStore
from app.rag.storage_models import RetrievedChunk
from tests.pdf_helpers import write_text_pdf


class FakeEmbedding:
    model="qwen3-embedding:0.6b"
    dimensions=1024
    query_instruction="Find evidence."
    def __init__(self):
        self.documents=[]
        self.queries=[]
        self.revision="revision-1"
        self.failure=None
        self.change_revision=False
        self.entered=None
        self.release=None
    async def get_revision(self):
        return self.revision
    async def embed_documents(self,texts):
        self.documents.append(texts)
        if self.entered is not None:
            self.entered.set()
            await self.release.wait()
        if self.failure:
            raise self.failure
        if self.change_revision:
            self.revision="revision-2"
        return [[1.0]+[0.0]*1023 for _ in texts]
    async def embed_query(self,question):
        self.queries.append(question)
        return [1.0]+[0.0]*1023


class FakeStore:
    def __init__(self):
        self.saved={}
        self.failure=None
        self.searches=[]
    async def save(self,info,prepared,vectors):
        assert Path(info.source_path).is_file()
        if self.failure:
            raise self.failure
        self.saved[info.paper_id]=(info,prepared,vectors)
    async def get_info(self,paper_id):
        return self.saved[paper_id][0] if paper_id in self.saved else None
    async def get(self,paper_id):
        pytest.fail("Questions must not load every page/chunk/vector")
    async def search(self,paper_id,vector,profile,*,top_k=5):
        self.searches.append((paper_id,profile,top_k))
        return [RetrievedChunk(paper_id=paper_id,chunk=c,score=0.8) for c in self.saved[paper_id][1].chunks[:top_k]]


class FakeAnswer:
    async def answer(self,paper_id,question,hits):
        from app.rag.evidence import build_evidence,verify_answer
        return verify_answer(paper_id,{"claims":[{"text":"Paper evidence.","evidence_ids":["E1"]}],
                                       "insufficient_evidence":False,"reason":None},build_evidence(paper_id,hits))


def pdf_bytes(tmp_path):
    path=tmp_path/"fixture.pdf"
    write_text_pdf(path,["Paper cover",None,"Baseline: 42 requests/s."])
    return path.read_bytes()


def setup_service(tmp_path,**settings):
    from app.rag.service import RagService
    embedding,store,traces=FakeEmbedding(),FakeStore(),InMemoryTraceStore()
    config=Settings(_env_file=None,rag_storage_dir=str(tmp_path/"stored"),**settings)
    service=RagService(store=store,embedding=embedding,answer=FakeAnswer(),trace_store=traces,settings=config)
    return service,embedding,store,traces


def test_import_then_question_preserve_sources_and_reuse_stored_vectors(tmp_path):
    data=pdf_bytes(tmp_path)
    async def run():
        service,embedding,store,traces=setup_service(tmp_path)
        result=await service.import_pdf(BytesIO(data),filename="../../evil.pdf",title="Cortex sample",
                                        request_id="req-import",trace_id="import-1")
        assert result.page_count==3 and result.chunk_count==2
        info,prepared,vectors=store.saved[result.paper_id]
        assert info.title=="Cortex sample"
        assert Path(info.source_path).parent==(tmp_path/"stored").resolve()
        assert Path(info.source_path).read_bytes()==data
        assert [p.page_number for p in prepared.pages]==[1,2,3]
        assert info.embedding.revision=="revision-1"
        answer=await service.ask(result.paper_id,"What is the baseline?",top_k=2,request_id="req-q",trace_id="question-1")
        assert answer.citations[0].paper_id==result.paper_id
        assert len(embedding.documents)==1 and len(embedding.queries)==1
        assert store.searches[0][2]==2
        imported=await traces.get("import-1")
        assert [s.name for s in imported.spans]==["rag.import","rag.admission","rag.upload","rag.pdf.extract",
                                                  "rag.text.prepare","rag.documents.embed","rag.paper.save"]
        assert all(s.status=="ok" for s in imported.spans)
        assert all(s.parent_span_id==imported.spans[0].span_id for s in imported.spans[1:])
        asked=await traces.get("question-1")
        assert [s.name for s in asked.spans]==["rag.ask","rag.admission","rag.paper.load","rag.question.embed",
                                             "rag.paper.search","rag.answer.generate"]
        assert asked.status=="ok"
    asyncio.run(run())


@pytest.mark.parametrize("limit", ["bytes","pages","text","chunks"])
def test_import_resource_limits_fail_without_embedding_or_database_write(tmp_path,limit):
    from app.rag.service_errors import RagLimitError
    config={"bytes":{"rag_max_file_bytes":10},"pages":{"rag_max_pages":1},
            "text":{"rag_max_text_chars":5},"chunks":{"rag_max_chunks":1}}[limit]
    data=pdf_bytes(tmp_path)
    async def run():
        service,embedding,store,traces=setup_service(tmp_path,**config)
        with pytest.raises(RagLimitError):
            await service.import_pdf(BytesIO(data),filename="paper.pdf",title=None,request_id="r",trace_id="limited")
        assert not embedding.documents and not store.saved
        assert not list((tmp_path/"stored").glob("*.pdf"))
        assert (await traces.get("limited")).status=="error"
    asyncio.run(run())


def test_invalid_pdf_and_all_blank_pdf_fail_with_trace_and_cleanup(tmp_path):
    from app.rag.pdf import PdfReadError
    from app.rag.preparation import NoUsableTextError
    blank=tmp_path/"blank.pdf"
    write_text_pdf(blank,[None])
    async def run():
        service,_,store,traces=setup_service(tmp_path)
        for data,error in [(b"not a PDF",PdfReadError),(blank.read_bytes(),NoUsableTextError)]:
            with pytest.raises(error):
                await service.import_pdf(BytesIO(data),filename="paper.pdf",title=None,request_id="r",trace_id=error.__name__)
            assert (await traces.get(error.__name__)).status=="error"
        assert not store.saved and not list((tmp_path/"stored").glob("*.pdf"))
    asyncio.run(run())


def test_model_revision_change_rejects_import_before_save(tmp_path):
    from app.rag.store import EmbeddingMismatchError
    data=pdf_bytes(tmp_path)
    async def run():
        service,embedding,store,traces=setup_service(tmp_path)
        embedding.change_revision=True
        with pytest.raises(EmbeddingMismatchError):
            await service.import_pdf(BytesIO(data),filename="paper.pdf",title=None,request_id="r",trace_id="revision")
        assert not store.saved and not list((tmp_path/"stored").glob("*.pdf"))
        assert (await traces.get("revision")).spans[-1].name=="rag.documents.embed"
    asyncio.run(run())


def test_store_failure_keeps_pdf_for_uncertain_commit_and_preserves_exception(tmp_path):
    data=pdf_bytes(tmp_path)
    async def run():
        service,_,store,traces=setup_service(tmp_path)
        error=RuntimeError("database write failed")
        store.failure=error
        with pytest.raises(RuntimeError) as caught:
            await service.import_pdf(BytesIO(data),filename="paper.pdf",title=None,request_id="r",trace_id="store-fail")
        assert caught.value is error
        assert len(list((tmp_path/"stored").glob("*.pdf")))==1
        assert (await traces.get("store-fail")).spans[0].attributes["source_file_retained"] is True
    asyncio.run(run())


def test_trace_failure_never_overrides_business_success_or_error(tmp_path):
    data=pdf_bytes(tmp_path)
    class BrokenTrace:
        async def save(self,trace):
            raise RuntimeError("trace unavailable")
    async def run():
        service,embedding,store,_=setup_service(tmp_path)
        service._trace_store=BrokenTrace()
        result=await service.import_pdf(BytesIO(data),filename="paper.pdf",title=None,request_id="r",trace_id="ok")
        assert result.paper_id in store.saved
        error=RuntimeError("embedding failed")
        embedding.failure=error
        with pytest.raises(RuntimeError) as caught:
            await service.import_pdf(BytesIO(data),filename="paper.pdf",title=None,request_id="r",trace_id="bad")
        assert caught.value is error
    asyncio.run(run())


def test_missing_paper_or_mismatched_model_does_not_generate_query(tmp_path):
    from app.rag.store import EmbeddingMismatchError,PaperNotFoundError
    data=pdf_bytes(tmp_path)
    async def run():
        service,embedding,_,traces=setup_service(tmp_path)
        with pytest.raises(PaperNotFoundError):
            await service.ask("missing","question",request_id="r",trace_id="missing")
        result=await service.import_pdf(BytesIO(data),filename="paper.pdf",title=None,request_id="r",trace_id="created")
        embedding.revision="changed"
        with pytest.raises(EmbeddingMismatchError):
            await service.ask(result.paper_id,"question",request_id="r",trace_id="mismatch")
        assert not embedding.queries
        assert (await traces.get("mismatch")).status=="error"
    asyncio.run(run())


def test_concurrency_wait_timeout_and_cancellation_release_capacity(tmp_path):
    from app.rag.service_errors import RagBusyError
    data=pdf_bytes(tmp_path)
    async def run():
        service,embedding,_,traces=setup_service(tmp_path,rag_max_concurrency=1,rag_queue_timeout_seconds=0.03)
        embedding.entered,embedding.release=asyncio.Event(),asyncio.Event()
        task=asyncio.create_task(service.import_pdf(BytesIO(data),filename="paper.pdf",title=None,request_id="r",trace_id="cancel"))
        await embedding.entered.wait()
        with pytest.raises(RagBusyError):
            await service.ask("anything","question",request_id="r",trace_id="busy")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert (await traces.get("cancel")).status=="cancelled"
        assert not list((tmp_path/"stored").glob("*.pdf"))
        embedding.entered=None
        result=await service.import_pdf(BytesIO(data),filename="paper.pdf",title=None,request_id="r",trace_id="after")
        assert result.paper_id
    asyncio.run(run())


def test_generated_file_collision_never_deletes_an_existing_source(tmp_path,monkeypatch):
    from app.rag import service as module
    data=pdf_bytes(tmp_path)
    existing=tmp_path/"stored"/"fixed-id.pdf"
    existing.parent.mkdir()
    existing.write_bytes(b"existing source")
    monkeypatch.setattr(module,"uuid4",lambda:"fixed-id")
    async def run():
        service,_,store,_=setup_service(tmp_path)
        with pytest.raises(FileExistsError):
            await service.import_pdf(BytesIO(data),filename="paper.pdf",title=None,request_id="r",trace_id="collision")
        assert existing.read_bytes()==b"existing source"
        assert not store.saved
    asyncio.run(run())
