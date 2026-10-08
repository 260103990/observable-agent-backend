import logging
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request

from app.api.routes.chat import router as chat_router
from app.api.routes.health import router as health_router
from app.api.routes.agent import router as agent_router


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


app = FastAPI(title="Observable Agent Backend")
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(agent_router)


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

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = (
        f"{process_time_ms:.2f}"
    )

    return response
