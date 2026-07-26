"""
Entry point: минимальный FastAPI-сервер вокруг сквозного пайплайна.

Запуск:
    uvicorn aqr.main:app --reload --port 8000

Endpoints:
    POST /pipeline/runs                  — стартовать run по свободному запросу
    GET  /pipeline/runs/{run_id}         — снимок событий
    GET  /pipeline/runs/{run_id}/stream  — SSE-лента событий
    WS   /chat/{session_id}              — двусторонний диалог с агентом
    GET  /health                         — liveness (всегда 200)
    GET  /health/ready                   — readiness (проверяет Postgres + MOEX)
"""
from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from aqr import __version__, tasks
from aqr.chat import chat_router
from aqr.chat.web import router as chat_web_router
from aqr.logging_config import setup_logging
from aqr.pipeline.api import router as pipeline_router

# Configure logging from AQR_LOG_JSON env var (default: human-readable)
setup_logging()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan: на shutdown дожидаемся фоновых задач (PERF-1)."""
    yield
    # До 30 секунд на завершение всех pipeline-тасков
    await tasks.drain(timeout=30.0)


app = FastAPI(
    title="AQR",
    description="Thin pipeline: natural-language goal -> validated hypotheses on MOEX",
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
    """Readiness probe: проверяет доступность Postgres и MOEX.

    Возвращает 503 если хотя бы одна зависимость недоступна. K8s не даёт
    трафик на под с degraded статусом.
    """
    from aqr.db import _async_session_factory

    checks: dict[str, str] = {}
    overall_ok = True

    # Postgres check
    try:
        from sqlalchemy import text
        async with _async_session_factory() as db:
            await asyncio.wait_for(db.execute(text("SELECT 1")), timeout=3.0)
        checks["postgres"] = "ok"
    except Exception as e:
        checks["postgres"] = f"down: {type(e).__name__}"
        overall_ok = False

    # MOEX check (синхронный HEAD в threadpool)
    def _moex_head() -> str:
        try:
            r = requests.head("https://iss.moex.com/iss/", timeout=5)
            if r.status_code == 200:
                return "ok"
            return f"down: HTTP {r.status_code}"
        except Exception as e:
            return f"down: {type(e).__name__}"

    try:
        moex_status = await asyncio.wait_for(
            asyncio.to_thread(_moex_head),
            timeout=6.0,
        )
    except (TimeoutError, Exception) as e:
        moex_status = f"down: {type(e).__name__}"
    checks["moex"] = moex_status
    if moex_status != "ok":
        overall_ok = False

    if not overall_ok:
        response.status_code = 503
    return {"status": "ready" if overall_ok else "degraded", **checks}
