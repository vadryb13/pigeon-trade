"""RegistryStore — асинхронный CRUD для Run, Hypothesis, Session."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import ChatMessage, Hypothesis, Run, Session


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

    # ── Chat history ─────────────────────────────────────────

    async def save_chat_message(
        self,
        session_id: str,
        role: str,
        content: str,
        meta: dict | None = None,
    ) -> ChatMessage:
        """Сохранить одно сообщение в историю чата сессии.

        Создаёт Session если отсутствует. Возвращает сохранённое сообщение.
        """
        await self.get_or_create_session(session_id)
        msg = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            meta=meta,
        )
        self._db.add(msg)
        await self._db.flush()
        return msg

    async def list_chat_history(
        self,
        session_id: str,
        limit: int = 200,
    ) -> list[ChatMessage]:
        """Последние сообщения сессии, отсортированы по created_at ASC.

        Tiebreaker по id UUID гарантирует детерминированный порядок при
        одинаковом created_at (что реально при microsecond-точности Postgres).
        """
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc())
            .limit(limit)
        )
        result = await self._db.execute(stmt)
        return list(result.scalars().all())

    # ── Run ──────────────────────────────────────────────────

    async def create_run(
        self,
        goal: str,
        session_id: str,
        status: str = "running",
        summary_metrics: dict | None = None,
        id: uuid.UUID | None = None,
    ) -> Run:
        run = Run(
            id=id,
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

    async def list_hypotheses_by_run(self, run_id: uuid.UUID) -> list[Hypothesis]:
        stmt = (
            select(Hypothesis)
            .where(Hypothesis.run_id == run_id)
            .order_by(Hypothesis.dsr.desc().nullslast())
        )
        result = await self._db.execute(stmt)
        return list(result.scalars().all())

    # ── Semantic search через pgvector ────────────────────────

    async def search_similar(
        self,
        embedding: list[float],
        threshold: float = 0.0,
        limit: int = 10,
    ) -> list[tuple[Hypothesis, float]]:
        """Поиск похожих гипотез по embedding через pgvector cosine distance.

        Returns:
            список кортежей (Hypothesis, similarity) отсортированный по убыванию
            similarity. `similarity = 1 - cosine_distance`. Гипотезы без
            embedding исключаются. Если `threshold > 0` — отсекаются ниже порога.
        """
        dist = Hypothesis.embedding.cosine_distance(embedding)
        sim_expr = (1 - dist).label("similarity")
        stmt = (
            select(Hypothesis, sim_expr)
            .where(Hypothesis.embedding.is_not(None))
            .order_by(dist.asc())
            .limit(limit * 2)  # берём больше, чтобы отсеять ниже порога
        )
        result = await self._db.execute(stmt)
        rows = list(result.all())
        if threshold > 0:
            rows = [(h, s) for h, s in rows if float(s) >= threshold]
        rows = rows[:limit]
        return [(h, float(s)) for h, s in rows]

    async def search_by_text(
        self,
        text: str,
        embedder: Any,
        limit: int = 10,
    ) -> list[tuple[Hypothesis, float]]:
        """Текст → embedding → search_similar.

        Args:
            text: произвольный запрос ("mean reversion на банках")
            embedder: инстанс `Embedder` (для lazy embedding через OpenAI/hash)
            limit: максимум результатов
        """
        emb = await embedder.embed(text)
        return await self.search_similar(emb, threshold=0.0, limit=limit)
