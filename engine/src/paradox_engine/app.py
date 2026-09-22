import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from paradox_engine.alchemy import repository
from paradox_engine.alchemy.service import AlchemyService
from paradox_engine.api.routers import alchemy, classpect, fraymotifs, health
from paradox_engine.api.rate_limit import IPRateLimiter
from paradox_engine.classpect.service import ClasspectService
from paradox_engine.classpect.thread_service import ClasspectThreadService
from paradox_engine.config import Settings, get_settings
from paradox_engine.fraymotifs.service import FraymotifService
from paradox_engine.llm import LLMClient
from paradox_engine.prompts.library import PromptLibrary
from paradox_engine.runtime import RuntimeResources

logger = logging.getLogger(__name__)


def _load_quiz(path: Path) -> list[dict]:
    try:
        with path.open(encoding="utf-8") as file:
            quiz = json.load(file)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required quiz file is missing: {path}") from exc
    if not isinstance(quiz, list):
        raise RuntimeError(f"Quiz file must contain a JSON array: {path}")
    return quiz


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        prompts = PromptLibrary.load(settings.prompts_directory)
        llm = LLMClient(settings.openrouter_api_key)
        class_quiz = _load_quiz(settings.class_quiz_filename)
        aspect_quiz = _load_quiz(settings.aspect_quiz_filename)
        classpect_service = ClasspectService(
            llm=llm,
            prompts=prompts,
            settings=settings,
        )
        app.state.resources = RuntimeResources(
            alchemy=AlchemyService(
                repository=repository,
                llm=llm,
                prompts=prompts,
                model=settings.alchemy_model,
            ),
            classpect=classpect_service,
            classpect_threads=ClasspectThreadService(
                llm=llm,
                prompts=prompts,
                settings=settings,
                classpect=classpect_service,
                class_quiz=class_quiz,
                aspect_quiz=aspect_quiz,
            ),
            fraymotifs=FraymotifService(
                llm=llm,
                prompts=prompts,
                model=settings.fraymotif_model,
            ),
            class_quiz=class_quiz,
            aspect_quiz=aspect_quiz,
        )
        try:
            yield
        finally:
            await repository.engine.dispose()

    app = FastAPI(
        lifespan=lifespan,
        root_path=settings.api_root_path,
        openapi_url="/openapi.json" if settings.enable_docs else None,
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url="/redoc" if settings.enable_docs else None,
    )
    app.state.rate_limiter = IPRateLimiter()
    if settings.parsed_cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.parsed_cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
        )

    @app.exception_handler(Exception)
    async def unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled request error", extra={"path": request.url.path})
        return JSONResponse(
            status_code=500, content={"detail": "Internal server error"}
        )

    app.include_router(health.router)
    app.include_router(classpect.router)
    app.include_router(alchemy.router)
    app.include_router(fraymotifs.router)
    return app


app = create_app()
