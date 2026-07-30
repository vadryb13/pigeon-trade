"""Explore API — REST endpoints for dynamic explore data."""
from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from aqr.registry.store import RegistryStore
from aqr.session import async_session_factory

router = APIRouter(prefix="/api/explore")


async def _hypotheses() -> list[dict]:
    async with async_session_factory() as db:
        store = RegistryStore(db)
        return await store.list_all_hypotheses()


async def _stats() -> dict:
    async with async_session_factory() as db:
        store = RegistryStore(db)
        return await store.get_explore_stats()


async def _activity(days: int = 7) -> list[dict]:
    async with async_session_factory() as db:
        store = RegistryStore(db)
        return await store.get_recent_activity(days=days)


async def _detail(hyp_id: str) -> dict | None:
    try:
        hyp_uuid = uuid.UUID(hyp_id)
    except ValueError:
        hyp_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, f"hyp-{hyp_id}")
    async with async_session_factory() as db:
        store = RegistryStore(db)
        return await store.get_hypothesis_detail(hyp_uuid)


@router.get("/hypotheses")
async def list_hypotheses(limit: int = 100, offset: int = 0):
    """Список гипотез для explore-таблицы."""
    try:
        h = await _hypotheses()
        return {"hypotheses": h[offset:offset + limit]}
    except Exception:
        return JSONResponse(
            status_code=503,
            content={"error": "Database unavailable", "hypotheses": []},
        )


@router.get("/hypotheses/{hyp_id}")
async def hypothesis_detail(hyp_id: str):
    """Детали гипотезы для notebook."""
    try:
        detail = await _detail(hyp_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="Hypothesis not found")
        return detail
    except HTTPException:
        raise
    except Exception:
        return JSONResponse(
            status_code=503,
            content={"error": "Database unavailable"},
        )


@router.get("/stats")
async def explore_stats():
    """Агрегатные метрики для stats-бара."""
    try:
        return await _stats()
    except Exception:
        return {"total_hypotheses": 0, "total_runs": 0, "approved": 0, "win_rate": 0}


@router.get("/activity")
async def activity_feed(days: int = 7):
    """Лента событий за N дней."""
    try:
        e = await _activity(days=days)
        return {"events": e}
    except Exception:
        return {"events": []}
