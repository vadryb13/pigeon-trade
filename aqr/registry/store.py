"""RegistryStore — асинхронный CRUD для Run, Hypothesis, Session."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import Hypothesis, Run, Session


class RegistryStore:
    """CRUD-операции над таблицами реестра гипотез."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    # ── Session ──────────────────────────────────────────────

    async def get_or_create_session(self, session_id: str) -> Session:
        sess = await self._db.get(Session, session_id)
        if sess is None:
            sess = Session(id=session_id)
            self._db.add(sess)
            await self._db.flush()
        return sess

    async def touch_session(self, session_id: str) -> None:
        sess = await self._db.get(Session, session_id)
        if sess is not None:
            sess.last_activity_at = datetime.now(UTC)
            await self._db.flush()

    # ── Run ──────────────────────────────────────────────────

    async def create_run(
        self,
        goal: str,
        session_id: str,
        status: str = "running",
        summary_metrics: dict | None = None,
    ) -> Run:
        run = Run(
            goal=goal,
            session_id=session_id,
            status=status,
            summary_metrics=summary_metrics,
        )
        self._db.add(run)
        await self._db.flush()
        return run

    async def get_run(self, run_id: uuid.UUID) -> Run | None:
        return await self._db.get(Run, run_id)

    async def list_runs_by_session(
        self, session_id: str, limit: int = 20, offset: int = 0
    ) -> list[Run]:
        stmt = (
            select(Run)
            .where(Run.session_id == session_id)
            .order_by(Run.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        result = await self._db.execute(stmt)
        return list(result.scalars().all())

    async def update_run_status(
        self, run_id: uuid.UUID, status: str, summary_metrics: dict | None = None
    ) -> None:
        run = await self._db.get(Run, run_id)
        if run is not None:
            run.status = status
            if summary_metrics is not None:
                run.summary_metrics = summary_metrics
            await self._db.flush()

    # ── Hypothesis ───────────────────────────────────────────

    async def create_hypothesis(
        self,
        run_id: uuid.UUID,
        family: str,
        ticker: str,
        config_json: dict,
        dsr: float | None = None,
        pbo: float | None = None,
        cpcv: float | None = None,
        sharpe: float | None = None,
        max_drawdown: float | None = None,
        is_valid: bool = False,
        embedding: list[float] | None = None,
    ) -> Hypothesis:
        hyp = Hypothesis(
            run_id=run_id,
            family=family,
            ticker=ticker,
            config_json=config_json,
            dsr=dsr,
            pbo=pbo,
            cpcv=cpcv,
            sharpe=sharpe,
            max_drawdown=max_drawdown,
            is_valid=is_valid,
            embedding=embedding,
        )
        self._db.add(hyp)
        await self._db.flush()
        return hyp

    async def create_hypotheses_bulk(self, hypotheses: list[Hypothesis]) -> None:
        self._db.add_all(hypotheses)
        await self._db.flush()

    async def list_hypotheses_by_run(self, run_id: uuid.UUID) -> list[Hypothesis]:
        stmt = (
            select(Hypothesis)
            .where(Hypothesis.run_id == run_id)
            .order_by(Hypothesis.dsr.desc().nullslast())
        )
        result = await self._db.execute(stmt)
        return list(result.scalars().all())

    async def get_top_valid_hypotheses(
        self, run_id: uuid.UUID, limit: int = 5
    ) -> list[Hypothesis]:
        stmt = (
            select(Hypothesis)
            .where(Hypothesis.run_id == run_id, Hypothesis.is_valid.is_(True))
            .order_by(Hypothesis.dsr.desc().nullslast())
            .limit(limit)
        )
        result = await self._db.execute(stmt)
        return list(result.scalars().all())

    # ── Batch save from pipeline result ──────────────────────

    async def save_pipeline_result(
        self,
        run_id: uuid.UUID,
        goal: str,
        session_id: str,
        status: str,
        summary_metrics: dict | None,
        hypotheses_data: list[dict[str, Any]],
    ) -> Run:
        """Атомарно сохраняет Run + все Hypothesis из результата пайплайна."""
        run = Run(
            id=run_id,
            goal=goal,
            session_id=session_id,
            status=status,
            summary_metrics=summary_metrics,
        )
        self._db.add(run)

        for h in hypotheses_data:
            self._db.add(Hypothesis(
                run_id=run_id,
                family=h["family"],
                ticker=h["ticker"],
                config_json=h.get("config_json", {}),
                dsr=h.get("dsr"),
                pbo=h.get("pbo"),
                cpcv=h.get("cpcv"),
                sharpe=h.get("sharpe"),
                max_drawdown=h.get("max_drawdown"),
                is_valid=h.get("is_valid", False),
                embedding=h.get("embedding"),
            ))

        await self._db.flush()
        return run
