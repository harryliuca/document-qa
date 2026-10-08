import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
from anyio import to_process
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException

from app.config import Settings
from app.ingestion import document_kind, parse_document, parse_questions
from app.middleware import RequestBoundary, log_event
from app.models import ClientFault, ProviderFault, QAResponse
from app.provider import OpenAIProvider, Provider
from app.service import QAService, embedding_token_count

STATIC = Path(__file__).parent / "static"


def create_app(settings: Settings | None = None, provider: Provider | None = None) -> FastAPI:
    config = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        for name in ("httpx", "httpcore", "httpx2", "httpcore2", "openai"):
            logging.getLogger(name).setLevel(logging.WARNING)
        owned = None
        active_provider = provider
        if active_provider is None and config.openai_api_key.get_secret_value():
            owned = OpenAIProvider(
                config.openai_api_key.get_secret_value(), config.model, config.answer_timeout
            )
            active_provider = owned
        app.state.service = QAService(active_provider, config) if active_provider else None
        app.state.parser_slots = anyio.CapacityLimiter(config.max_concurrent_requests)
        # Warm tokenizer once; Docker preloads its cache to avoid runtime downloads.
        await anyio.to_thread.run_sync(embedding_token_count, ["warmup"])
        try:
            yield
        finally:
            if owned:
                await owned.close()

    app = FastAPI(
        title="Document QA", version="0.5.0", lifespan=lifespan, docs_url=None, redoc_url=None
    )
    app.add_middleware(RequestBoundary, settings=config)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.exception_handler(ClientFault)
    async def client_error(request: Request, exc: ClientFault):
        return JSONResponse(
            {
                "error": {"code": exc.code, "message": exc.message},
                "request_id": request.state.request_id,
            },
            status_code=exc.status,
        )

    @app.exception_handler(ProviderFault)
    async def provider_error(request: Request, exc: ProviderFault):
        status = (
            503 if exc.code in {"provider_limit", "provider_auth", "provider_unavailable"} else 502
        )
        return JSONResponse(
            {
                "error": {"code": exc.code, "message": exc.message},
                "request_id": request.state.request_id,
            },
            status_code=status,
        )

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        return JSONResponse(
            {
                "error": {"code": "invalid_request", "message": "Invalid request or upload."},
                "request_id": request.state.request_id,
            },
            status_code=exc.status_code,
        )

    @app.get("/", include_in_schema=False)
    async def home():
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    async def health():
        return {"status": "ok", "ai_configured": app.state.service is not None}

    @app.post(
        "/api/answers",
        response_model=QAResponse,
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "multipart/form-data": {
                        "schema": {
                            "type": "object",
                            "required": ["document", "questions"],
                            "properties": {
                                "document": {"type": "string", "format": "binary"},
                                "questions": {"type": "string", "format": "binary"},
                            },
                        }
                    }
                },
            },
        },
    )
    async def answer_questions(request: Request):
        started = time.monotonic()
        async with request.form(max_files=2, max_fields=0) as form:
            if sorted(form.keys()) != ["document", "questions"] or len(form.multi_items()) != 2:
                raise ClientFault(
                    "missing_files", "Upload exactly two files: document and questions."
                )
            document, questions_file = form["document"], form["questions"]
            if not isinstance(document, UploadFile) or not isinstance(questions_file, UploadFile):
                raise ClientFault("missing_files", "Both fields must be file uploads.")
            kind = document_kind(document.filename or "")
            raw_questions = await questions_file.read(config.max_questions_bytes + 1)
            questions = parse_questions(raw_questions, config)
            raw_document = await document.read(config.max_document_bytes + 1)
            if len(raw_document) > config.max_document_bytes:
                raise ClientFault("document_too_large", "Document exceeds 10 MB.", 413)
        limits = {
            name: getattr(config, name)
            for name in ("max_document_bytes", "max_pages", "max_chars", "max_chunks")
        }
        try:
            async with asyncio.timeout(config.parser_timeout):
                chunks = await to_process.run_sync(
                    parse_document,
                    raw_document,
                    kind,
                    limits,
                    cancellable=True,
                    limiter=app.state.parser_slots,
                )
        except TimeoutError:
            raise ClientFault(
                "parse_timeout", "Document parsing timed out. Try a smaller PDF.", 422
            ) from None
        service = app.state.service
        if service is None:
            raise ClientFault("not_configured", "Server requires an OpenAI API key.", 503)
        results, usage = await service.run(chunks, questions)
        duration = round((time.monotonic() - started) * 1000)
        log_event(
            "qa_complete",
            request_id=request.state.request_id,
            chunks=len(chunks),
            questions=len(questions),
            answered=sum(r.status == "answered" for r in results),
            not_found=sum(r.status == "not_found" for r in results),
            errors=sum(r.status == "error" for r in results),
            duration_ms=duration,
            **usage.model_dump(),
        )
        return QAResponse(
            request_id=request.state.request_id,
            results=results,
            usage=usage,
            duration_ms=duration,
            document_chunks=len(chunks),
        )

    return app
