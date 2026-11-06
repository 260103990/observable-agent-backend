"""Paper ingestion and grounded questions, with safe public error messages."""
from typing import Annotated, Any

import asyncpg
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from app.api.rag_dependencies import get_rag_service, get_rag_trace_id
from app.rag.answer import (AnswerConfigurationError, AnswerConnectionError,
                            AnswerResponseError, AnswerTimeoutError)
from app.rag.embedding import (EmbeddingConnectionError, EmbeddingResponseError,
                              EmbeddingTimeoutError)
from app.rag.evidence import CitationValidationError
from app.rag.pdf import PdfReadError
from app.rag.preparation import NoUsableTextError
from app.rag.service_errors import (RagBusyError, RagInputError, RagLimitError,
                                    RagUnavailableError)
from app.rag.store import (EmbeddingMismatchError, PaperAlreadyExistsError,
                           PaperNotFoundError)
from app.schemas.papers import (PaperImportResponse, PaperQuestionRequest,
                                 PaperQuestionResponse)


router = APIRouter(prefix="/papers", tags=["papers"])
TraceId = Annotated[str, Depends(get_rag_trace_id)]
RagServiceDependency = Annotated[Any, Depends(get_rag_service)]

_ERROR_RESPONSES = (
    ((RagLimitError,), 413, "Paper exceeds the configured limit"),
    ((PaperNotFoundError,), 404, "Paper was not found"),
    ((PaperAlreadyExistsError, EmbeddingMismatchError), 409, "Paper data conflicts with this operation"),
    ((EmbeddingTimeoutError, AnswerTimeoutError), 504, "RAG provider request timed out"),
    ((EmbeddingConnectionError, AnswerConnectionError, RagBusyError,
      RagUnavailableError, AnswerConfigurationError, asyncpg.PostgresError, OSError),
     503, "RAG service is unavailable"),
    ((EmbeddingResponseError, AnswerResponseError, CitationValidationError),
     502, "RAG provider returned an invalid response"),
    ((RagInputError, PdfReadError, NoUsableTextError), 400, "Invalid paper input"),
)
_HANDLED_ERRORS = tuple(error for errors, _, _ in _ERROR_RESPONSES for error in errors)


def _public_error(error: Exception) -> HTTPException:
    for errors, status_code, detail in _ERROR_RESPONSES:
        if isinstance(error, errors):
            return HTTPException(status_code=status_code, detail=detail)
    raise TypeError("Unhandled RAG error type")


@router.post("", status_code=201, response_model=PaperImportResponse)
async def import_paper(
    request: Request,
    trace_id: TraceId,
    service: RagServiceDependency,
    file: Annotated[UploadFile, File()],
    title: Annotated[str | None, Form()] = None,
) -> PaperImportResponse:
    try:
        result = await service.import_pdf(
            file.file, file.filename or "upload.pdf", title,
            request_id=request.state.request_id, trace_id=trace_id,
        )
    except _HANDLED_ERRORS as error:
        raise _public_error(error) from None
    finally:
        await file.close()
    return PaperImportResponse(paper_id=result.paper_id, title=result.title,
                               page_count=result.page_count, chunk_count=result.chunk_count,
                               trace_id=trace_id)


@router.post("/{paper_id}/questions", response_model=PaperQuestionResponse)
async def ask_paper(
    paper_id: str,
    payload: PaperQuestionRequest,
    request: Request,
    trace_id: TraceId,
    service: RagServiceDependency,
) -> PaperQuestionResponse:
    try:
        result = await service.ask(
            paper_id, payload.question, top_k=payload.top_k,
            request_id=request.state.request_id, trace_id=trace_id,
        )
    except _HANDLED_ERRORS as error:
        raise _public_error(error) from None
    return PaperQuestionResponse(**result.model_dump(), trace_id=trace_id)
