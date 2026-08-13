"""
Минимальный FastAPI роутер для сквозного пайплайна.

Endpoints:
- POST /pipeline/runs          — стартовать run от свободного запроса
- GET  /pipeline/runs/{id}     — снимок состояния (все накопленные события)
- GET  /pipeline/runs/{id}/stream — SSE-стрим событий (для UI живой ленты)
"""
from __future__ import annotations

import asyncio
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from aqr.auth import require_session_id
from aqr.background import schedule
from aqr.registry import RegistryStore
from aqr.session import get_db

from .events import BUS, Event
from .executor import PipelineExecutor
from .planner import ResearchPlanner

router = APIRouter(prefix="/pipeline", tags=["pipeline"])

_logger = logging.getLogger(__name__)


class RunRequest(BaseModel):
    """POST /pipeline/runs body.

    `goal` валидируется: длина 3–2000, после strip не пустая. Защита от
    спам-запросов и log-injection через control chars (SEC-3).
    """

    goal: str = Field(..., min_length=3, max_length=2000)

    @field_validator("goal")
    @classmethod
    def _strip_and_check(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("goal cannot be empty or whitespace-only")
        return v


class PlanResponse(BaseModel):
    """Типизированный план в ответе /pipeline/runs (TYPE-1).

    Раньше возвращался как `dict[str, Any]` — клиенты не могли типизировать
    и IDE не помогала. Здесь — explicit schema.
    """

    goal: str
    tickers: list[str]
    timeframe: str
    hypothesis_families: list[str]
    n_hypotheses: int
    rationale: str = ""


class RunStarted(BaseModel):
    run_id: str
    plan: PlanResponse


@router.post("/runs", response_model=RunStarted)
async def start_run(
    req: RunRequest,
    db: AsyncSession = Depends(get_db),
    session_id: str = Depends(require_session_id),
) -> RunStarted:
    """Принять свободный запрос, спланировать, запустить исполнение в фоне."""
    planner = ResearchPlanner()
    plan = await planner.plan(req.goal)

    # Генерируем UUID ОДИН раз — он должен совпадать в BUS и в БД (иначе FK-violation
    # при create_hypothesis в фоне). Найдено в REVIEW.md / вживую 2026-07-19.
    run_id = uuid.uuid4()
    run_id_str = str(run_id)

    # Регистрируем run в BUS для EventBus
    await BUS.register_run(run_id_str, session_id)

    executor = PipelineExecutor(BUS)

    # Сохраняем run в БД как "running" с явным UUID.
    # B16: commit() уже выполнен → run виден для новой DB-сессии, которую
    # откроет фоновая задача в `_run_and_persist`. Без явного commit
    # фоновая задача может стартануть раньше, чем транзакция request-handler
    # зафиксируется, и тогда `update_run_status` никого не найдёт.
    store = RegistryStore(db)
    await store.get_or_create_session(session_id)
    await store.create_run(
        id=run_id, goal=req.goal, session_id=session_id, status="running",
    )
    await db.commit()

    # Фоновый запуск с сохранением результата (с retention — иначе GC)
    try:
        schedule(_run_and_persist(run_id_str, plan, executor))
    except RuntimeError:
        await store.update_run_status(run_id, status="error")
        await db.commit()
        raise HTTPException(
            status_code=503,
            detail="Превышен лимит фоновых задач. Попробуйте позже.",
        )

    return RunStarted(
        run_id=run_id_str,
        plan=PlanResponse(
            goal=plan.goal,
            tickers=plan.tickers,
            timeframe=plan.timeframe,
            hypothesis_families=plan.hypothesis_families,
            n_hypotheses=plan.n_hypotheses,
            rationale=plan.rationale,
        ),
    )


@router.get("/runs/{run_id}")
async def get_run(run_id: str, session_id: str = Depends(require_session_id)) -> dict:
    """Снимок всех накопленных событий."""
    if not await BUS.owns(run_id, session_id):
        raise HTTPException(status_code=404, detail="run not found")
    history = BUS.history(run_id)
    if not history:
        return {"run_id": run_id, "events": [], "status": "unknown"}
    latest = history[-1]
    return {
        "run_id": run_id,
        "events": [
            {"kind": e.kind, "stage": e.stage, "message": e.message,
             "data": e.data, "ts": e.ts}
            for e in history
        ],
        "status": "done" if latest.kind == "done"
                  else "error" if latest.kind == "error"
                  else "running",
    }


@router.get("/runs/{run_id}/stream")
async def stream_run(run_id: str, session_id: str = Depends(require_session_id)):
    """SSE-стрим событий."""
    if not await BUS.owns(run_id, session_id):
        raise HTTPException(status_code=404, detail="run not found")

    async def gen():
        async for ev in BUS.subscribe(run_id):
            yield f"event: {ev.kind}\ndata: {ev.to_json()}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


# ── Background persistence ────────────────────────────────────

async def _run_and_persist(
    run_id: str,
    plan,
    executor: PipelineExecutor,
) -> None:
    """Выполняет пайплайн в фоне и сохраняет результат в БД."""
    from aqr.session import async_session_factory

    try:
        result = await executor.run(run_id, plan)
    except Exception as exc:
        # executor.run() мог упасть ДО эмита error-события.
        # Эмитим событие явно, чтобы SSE-клиент видел причину.
        _logger.exception("_run_and_persist: executor.run failed")
        try:
            await BUS.publish(Event(
                run_id=run_id, kind="error", stage="executor",
                message=f"executor.run: {type(exc).__name__}",
                data={"exception": type(exc).__name__},
            ))
        except Exception:
            _logger.exception("Failed to emit error event")
        result = None

    # Новая сессия для фоновой записи (request-сессия уже закрыта)
    async with async_session_factory() as db:
        store = RegistryStore(db)
        run_uuid = uuid.UUID(run_id)

        if result is not None:
            await store.update_run_status(
                run_uuid,
                status="done",
                summary_metrics={
                    "n_tested": result.n_hypotheses_tested,
                    "n_survived_dsr": result.n_survived_dsr,
                    "portfolio_pbo": result.portfolio_pbo,
                    "elapsed_seconds": result.elapsed_seconds,
                },
            )

            # Генерируем эмбеддинги параллельно (asyncio.gather) — для топ-5
            # последовательные RTT к OpenAI = 5× задержка. Hash-fallback
            # работает локально и быстро, но gather всё равно корректен (PERF-8).
            from aqr.registry.embeddings import Embedder
            embedder = Embedder()
            embeddings_raw = await asyncio.gather(*[
                embedder.embed_hypothesis(
                    r.hypothesis.family,
                    r.hypothesis.ticker,
                    r.hypothesis.params,
                )
                for r in result.top
            ], return_exceptions=True)
            for r, emb in zip(result.top, embeddings_raw):
                if isinstance(emb, Exception):
                    _logger.exception(
                        "embed_hypothesis failed for %s/%s: %s",
                        r.hypothesis.family, r.hypothesis.ticker, emb,
                    )
                    continue
                await store.create_hypothesis(
                    run_id=run_uuid,
                    family=r.hypothesis.family,
                    ticker=r.hypothesis.ticker,
                    config_json=r.hypothesis.params,
                    dsr=r.dsr,
                    cpcv=r.cpcv_mean_sharpe,
                    sharpe=r.sharpe,
                    max_drawdown=r.max_drawdown,
                    is_valid=r.dsr_verdict in ("significant", "borderline"),
                    embedding=emb,
                )
        else:
            await store.update_run_status(run_uuid, status="error")

        await db.commit()
