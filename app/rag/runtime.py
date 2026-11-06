"""Explicit RAG startup: validate existing schema, then assemble borrowed resources."""
from app.rag.service_errors import RagUnavailableError


async def build_rag_service(*, pool, http, trace_store, settings):
    if settings.ollama_embedding_dimensions != 1024:
        raise RagUnavailableError("RAG storage schema requires 1024-dimensional embeddings")
    ready = await pool.fetchval("""
        SELECT to_regclass('public.rag_papers') IS NOT NULL
           AND to_regclass('public.rag_pages') IS NOT NULL
           AND to_regclass('public.rag_chunks') IS NOT NULL
           AND to_regclass('public.traces') IS NOT NULL
           AND to_regclass('public.spans') IS NOT NULL
           AND EXISTS (SELECT 1 FROM pg_extension WHERE extname='vector')
    """)
    if not ready:
        raise RagUnavailableError("RAG requires initialized paper/vector and Trace tables")
    from app.rag.answer import DeepSeekAnswerClient
    from app.rag.embedding import OllamaEmbeddingClient
    from app.rag.service import RagService
    from app.rag.store import PostgresPaperStore
    return RagService(store=PostgresPaperStore(pool),
                      embedding=OllamaEmbeddingClient.from_settings(settings,client=http),
                      answer=DeepSeekAnswerClient.from_settings(settings,client=http),
                      trace_store=trace_store,settings=settings)
