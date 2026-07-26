"""
Entry point: FastAPI-сервер вокруг сквозного пайплайна.

Запуск:
    uvicorn aqr.main:app --reload --port 8000

Endpoints:
    POST /pipeline/runs                  — стартовать run по свободному запросу
    GET  /pipeline/runs/{run_id}         — снимок событий
    GET  /pipeline/runs/{run_id}/stream  — SSE-лента событий
    WS   /chat/{token}                   — двусторонний диалог с агентом
    GET  /chat/{token}/settings          — форма настроек сессии (LLM/Invest keys)
    POST /chat/{token}/settings          — сохранение credentials в session_settings
    GET  /chat                           — HTML-страница чата
    GET  /chat/new?session_id=...        — выпуск HMAC-токена
    GET  /health                         — liveness (всегда 200)
    GET  /health/ready                   — readiness (Postgres + auto-provision)

Startup validation:
    `lifespan` вызывает `validate_runtime()` ДО `yield`. Если обязательные
    env отсутствуют или Postgres не поднимается — FastAPI не стартует.
"""
from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from aqr import __version__, tasks
from aqr.chat import chat_router
from aqr.chat.web import router as chat_web_router
from aqr.logging_config import setup_logging
from aqr.pipeline.api import router as pipeline_router
from aqr.startup import validate_runtime

# Configure logging from AQR_LOG_JSON env var (default: human-readable)
setup_logging()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan: на startup валидируем runtime, на shutdown дожидаемся задач."""
    await validate_runtime()
    try:
        yield
    finally:
        await tasks.drain(timeout=30.0)


app = FastAPI(
    title="AQR",
    description="Thin pipeline: natural-language goal -> validated hypotheses on MOEX via T-Invest",
    version=__version__,
    lifespan=lifespan,
)

_ALLOWED_ORIGINS = os.getenv(
    "AQR_ALLOWED_ORIGINS",
    "*",  # dev default — для прода AQR_ALLOWED_ORIGINS=https://app.example.com
).split(",")
_ALLOWED_METHODS = os.getenv(
    "AQR_ALLOWED_METHODS",
    "GET,POST",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _ALLOWED_ORIGINS],
    allow_methods=[m.strip() for m in _ALLOWED_METHODS],
    allow_headers=["*"],
)

app.include_router(pipeline_router)
app.include_router(chat_router)
app.include_router(chat_web_router)


@app.get("/health")
async def health() -> dict:
    """Liveness probe: всегда 200, если процесс жив. K8s не убивает под."""
    return {"status": "ok", "version": __version__}


@app.get("/health/ready")
async def health_ready(response: Response) -> dict:
    """Readiness probe: проверяет валидность runtime.

    Возвращает 503 если хотя бы одна проверка упала. K8s не даёт
    трафик на под с degraded статусом.
    """
    try:
        result = await validate_runtime()
        return result
    except RuntimeError as e:
        response.status_code = 503
        return {"status": "degraded", "error": str(e)}
