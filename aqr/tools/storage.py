"""Storage tools: get_run, compare_runs, list_runs.

Обёртки над RegistryStore для использования агентом.
"""

from __future__ import annotations

import uuid
from typing import Any

from ..db import _async_session_factory
from ..registry import RegistryStore


async def get_run(run_id: str) -> dict[str, Any] | None:
    """Получить детали прогона по run_id."""
    try:
        rid = uuid.UUID(run_id)
    except (ValueError, AttributeError):
        return {"error": f"Invalid run_id: {run_id!r}"}
    async with _async_session_factory() as db:
        store = RegistryStore(db)
        run = await store.get_run(rid)
        if run is None:
            return None
        hyps = await store.list_hypotheses_by_run(rid)
        return {
            "id": str(run.id),
            "goal": run.goal,
            "session_id": run.session_id,
            "status": run.status,
            "summary_metrics": run.summary_metrics,
            "created_at": run.created_at.isoformat(),
            "hypotheses": [
                {
                    "id": str(h.id),
                    "family": h.family,
                    "ticker": h.ticker,
                    "config_json": h.config_json,
                    "sharpe": h.sharpe,
                    "dsr": h.dsr,
                    "cpcv": h.cpcv,
                    "max_drawdown": h.max_drawdown,
                    "is_valid": h.is_valid,
                }
                for h in hyps
            ],
        }


async def compare_runs(
    run_id_a: str,
    run_id_b: str,
) -> dict[str, Any]:
    """Сравнить два прогона по ключевым метрикам."""
    try:
        rid_a = uuid.UUID(run_id_a)
        rid_b = uuid.UUID(run_id_b)
    except (ValueError, AttributeError):
        return {"error": f"Invalid run_id: {run_id_a!r} or {run_id_b!r}"}
    async with _async_session_factory() as db:
        store = RegistryStore(db)
        run_a = await store.get_run(rid_a)
        run_b = await store.get_run(rid_b)

        if run_a is None or run_b is None:
            return {"error": "Один или оба прогона не найдены"}

        ma = run_a.summary_metrics or {}
        mb = run_b.summary_metrics or {}

        return {
            "run_a": {
                "id": str(run_a.id),
                "goal": run_a.goal,
                "n_tested": ma.get("n_tested"),
                "n_survived_dsr": ma.get("n_survived_dsr"),
                "portfolio_pbo": ma.get("portfolio_pbo"),
                "elapsed_seconds": ma.get("elapsed_seconds"),
            },
            "run_b": {
                "id": str(run_b.id),
                "goal": run_b.goal,
                "n_tested": mb.get("n_tested"),
                "n_survived_dsr": mb.get("n_survived_dsr"),
                "portfolio_pbo": mb.get("portfolio_pbo"),
                "elapsed_seconds": mb.get("elapsed_seconds"),
            },
            "delta": {
                "n_survived_dsr": (
                    (mb.get("n_survived_dsr") or 0) - (ma.get("n_survived_dsr") or 0)
                ),
                "portfolio_pbo": (
                    round((mb.get("portfolio_pbo") or 0) - (ma.get("portfolio_pbo") or 0), 3)
                ),
            },
        }


async def list_runs(
    session_id: str = "default",
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Список последних прогонов в сессии."""
    async with _async_session_factory() as db:
        store = RegistryStore(db)
        runs = await store.list_runs_by_session(session_id, limit=limit)
        return [
            {
                "id": str(r.id),
                "goal": r.goal,
                "status": r.status,
                "summary_metrics": r.summary_metrics,
                "created_at": r.created_at.isoformat(),
            }
            for r in runs
        ]


async def search_similar_hypotheses(
    text: str,
    threshold: float = 0.7,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Поиск семантически похожих гипотез по текстовому запросу.

    Args:
        text: произвольное описание гипотезы ("momentum SMA5/50 on SBER")
        threshold: порог cosine similarity (0.0..1.0). По умолчанию 0.7.
        limit: максимум результатов.

    Returns:
        список {family, ticker, similarity, dsr, sharpe, run_id}. Пустой,
        если совпадений нет.

    Raises:
        RuntimeError: если БД недоступна или LLM/embeddings не настроены.
    """
    from ..registry.embeddings import Embedder

    embedder = Embedder()
    async with _async_session_factory() as db:
        store = RegistryStore(db)
        results = await store.search_by_text(text, embedder, limit=limit)
    return [
        {
            "family": h.family,
            "ticker": h.ticker,
            "similarity": round(sim, 3),
            "dsr": h.dsr,
            "sharpe": h.sharpe,
            "is_valid": h.is_valid,
            "run_id": str(h.run_id),
        }
        for h, sim in results
        if sim >= threshold
    ]


async def find_duplicates(
    text: str,
    threshold: float = 0.92,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Найти уже-проверенные гипотезы, похожие на данный текст.

    Удобно вызывать перед запуском прогона: если список непустой — стоит
    сменить параметры, чтобы не дублировать прошлые исследования.

    Args:
        text: описание новой гипотезы
        threshold: порог (по умолчанию 0.92 — высокий, exact-ish)
        limit: максимум результатов

    Returns:
        список {family, ticker, similarity, previous_dsr, previous_sharpe, run_id}.
    """
    return await search_similar_hypotheses(
        text=text, threshold=threshold, limit=limit
    )
