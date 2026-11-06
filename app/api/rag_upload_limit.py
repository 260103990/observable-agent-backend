"""Bound the multipart HTTP body before an upload can exhaust temporary storage."""
from fastapi import HTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.config import get_settings


class _UploadTooLarge(HTTPException):
    def __init__(self):
        super().__init__(status_code=413, detail="Paper upload exceeds the configured limit")


class RagUploadLimitMiddleware:
    def __init__(self, app: ASGIApp, *, settings=None):
        self.app = app
        self.settings = settings

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (scope["type"] != "http" or scope.get("method") != "POST"
                or scope.get("path", "").rstrip("/") != "/papers"):
            await self.app(scope, receive, send)
            return

        settings = self.settings if self.settings is not None else get_settings()
        limit = settings.rag_max_file_bytes + 65536
        for name, value in scope.get("headers", []):
            if name.lower() == b"content-length":
                try:
                    too_large = int(value) > limit
                except ValueError:
                    too_large = False
                if too_large:
                    await self._reject(scope, receive, send)
                    return

        received_bytes = 0
        rejected = False

        async def limited_receive() -> Message:
            nonlocal received_bytes
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > limit:
                    raise _UploadTooLarge()
            return message

        async def limited_send(message: Message) -> None:
            nonlocal rejected
            if received_bytes > limit:
                # BaseHTTPMiddleware can wrap a receive exception in an
                # ExceptionGroup, which FastAPI translates into HTTP 400.
                # Own the public response once this middleware detected overflow.
                if not rejected:
                    rejected = True
                    await self._reject(scope, receive, send)
                return
            await send(message)

        try:
            await self.app(scope, limited_receive, limited_send)
        except Exception:
            if received_bytes <= limit:
                raise
            if not rejected:
                await self._reject(scope, receive, send)

    @staticmethod
    async def _reject(scope: Scope, receive: Receive, send: Send) -> None:
        response = JSONResponse(status_code=413,
                                content={"detail": "Paper upload exceeds the configured limit"})
        await response(scope, receive, send)
