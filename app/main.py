import logging
from time import perf_counter
from uuid import uuid4
from contextlib import asynccontextmanager

import asyncpg
import httpx
from fastapi import FastAPI, Request

from app.api.routes.chat import router as chat_router
from app.api.routes.health import router as health_router
from app.api.routes.agent import router as agent_router
from app.api.routes.agent_loop import router as agent_loop_router
from app.api.routes.traces import router as traces_router
from app.config import get_settings
from app.observability.store import PostgresTraceStore
from app.api.routes.papers import router as papers_router
from app.api.rag_upload_limit import RagUploadLimitMiddleware
from app.rag.runtime import build_rag_service
from app.rag.service_errors import RagUnavailableError
from app.tools.search_cache import cache_from_settings


request_logger = logging.getLogger("app.request")
request_logger.setLevel(logging.INFO)
request_logger.propagate = False

if not request_logger.handlers:
    request_handler = logging.StreamHandler()

    request_handler.setFormatter(
    logging.Formatter(
        "%(levelname)s "
        "%(name)s "
        "%(message)s "
        "request_id=%(request_id)s "
        "method=%(method)s "
        "path=%(path)s "
        "status_code=%(status_code)s "
        "process_time_ms=%(process_time_ms).2f "
        "exception_type=%(exception_type)s",
        defaults={
            "exception_type": "-",
        },
    )
)

    request_logger.addHandler(request_handler)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.startup_complete = False
    settings = get_settings()
    cache = cache_from_settings(settings) if settings.web_search_cache_enabled else None
    if cache is not None:
        app.state.search_cache = cache
    try:
        async with _database_lifespan(app):
            app.state.startup_complete = True
            try:
                yield
            finally:
                # Become unready before database/HTTP resource cleanup begins.
                app.state.startup_complete = False
    finally:
        app.state.startup_complete = False
        if cache is not None:
            cache.clear()
            del app.state.search_cache


@asynccontextmanager
async def _database_lifespan(app: FastAPI):
    settings = get_settings()
    if settings.database_url is None:
        if settings.rag_enabled:
            raise RagUnavailableError("RAG_ENABLED requires DATABASE_URL")
        yield
        return
    pool = await asyncpg.create_pool(dsn=settings.database_url)
    app.state.db_pool = pool
    app.state.trace_store = PostgresTraceStore(pool)
    try:
        if settings.rag_enabled:
            connections = max(4, settings.rag_max_concurrency * 2)
            async with httpx.AsyncClient(trust_env=False, limits=httpx.Limits(
                max_connections=connections, max_keepalive_connections=connections,
            )) as http:
                app.state.rag_service = await build_rag_service(
                    pool=pool, http=http, trace_store=app.state.trace_store, settings=settings,
                )
                try:
                    yield
                finally:
                    del app.state.rag_service
        else:
            yield
    finally:
        await pool.close()
        del app.state.trace_store
        del app.state.db_pool


app = FastAPI(
    title="Observable Agent Backend",
    lifespan=lifespan,
)
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(agent_router)
app.include_router(agent_loop_router)
app.include_router(traces_router)
app.include_router(papers_router)
app.add_middleware(RagUploadLimitMiddleware)


@app.middleware("http")
async def add_request_context(
    request: Request,
    call_next,
):
    start_time = perf_counter()

    request_id = (
        request.headers.get("X-Request-ID")
        or str(uuid4())
    )

    request.state.request_id = request_id

    try:
        response = await call_next(request)
    except Exception as exc:
        process_time_ms = (
            perf_counter() - start_time
        ) * 1000

        request_logger.exception(
            "request_failed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": 500,
                "exception_type": type(exc).__name__,
                "process_time_ms": process_time_ms,
            },
        )

        raise

    process_time_ms = (
        perf_counter() - start_time
    ) * 1000

    request_logger.info(
        "request_completed",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "process_time_ms": process_time_ms,
        },
    )
    trace_id = getattr(
        request.state,
        "trace_id",
        None,
    )

    if trace_id is not None:
        response.headers["X-Trace-ID"] = trace_id

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = (
        f"{process_time_ms:.2f}"
    )

    return response
