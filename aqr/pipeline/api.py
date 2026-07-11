"""
Минимальный FastAPI роутер для сквозного пайплайна.

Endpoints:
- POST /pipeline/runs          — стартовать run от свободного запроса
- GET  /pipeline/runs/{id}     — снимок состояния (все накопленные события)
- GET  /pipeline/runs/{id}/stream — SSE-стрим событий (для UI живой ленты)
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from aqr.db import get_db
from aqr.registry import RegistryStore

from .events import BUS
from .executor import PipelineExecutor
from .planner import ChatPlanner


router = APIRouter(prefix="/pipeline", tags=["pipeline"])


class RunRequest(BaseModel):
    goal: str


class RunStarted(BaseModel):
    run_id: str
    plan: dict[str, Any]


@router.post("/runs", response_model=RunStarted)
async def start_run(
    req: RunRequest,
    db: AsyncSession = Depends(get_db),
) -> RunStarted:
    """Принять свободный запрос, спланировать, запустить исполнение в фоне."""
    planner = ChatPlanner()
    plan = planner.plan(req.goal)

    run_id = BUS.new_run()
    executor = PipelineExecutor(BUS)

    # Сохраняем run в БД как "running"
    store = RegistryStore(db)
    run_uuid = uuid.UUID(run_id)
    await store.get_or_create_session("default")
    await store.create_run(goal=req.goal, session_id="default", status="running")
    await db.commit()

    # Фоновый запуск с сохранением результата
    asyncio.create_task(_run_and_persist(run_id, plan, executor))

    return RunStarted(run_id=run_id, plan={
        "goal": plan.goal,
        "tickers": plan.tickers,
        "timeframe": plan.timeframe,
        "hypothesis_families": plan.hypothesis_families,
        "n_hypotheses": plan.n_hypotheses,
        "rationale": plan.rationale,
    })


@router.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict:
    """Снимок всех накопленных событий."""
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
async def stream_run(run_id: str):
    """SSE-стрим событий."""

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
    from aqr.db import _async_session_factory

    try:
        result = await executor.run(run_id, plan)
        status = "done"
    except Exception:
        # executor.run() уже эмитит error-событие, сохраняем статус
        result = None
        status = "error"

    # Новая сессия для фоновой записи (request-сессия уже закрыта)
    async with _async_session_factory() as db:
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

            # Сохраняем гипотезы
            for r in result.top:
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
                )
        else:
            await store.update_run_status(run_uuid, status="error")

        await db.commit()
