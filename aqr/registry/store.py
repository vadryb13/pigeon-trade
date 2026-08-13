"""RegistryStore — асинхронный CRUD для Run, Hypothesis, Session."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..crypto import decrypt_str, encrypt_str
from .models import ChatMessage, Hypothesis, Run, Session, SessionSettings


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

        Берём последние limit через DESC + reversed, чтобы вернуть ASC-порядок.
        Tiebreaker по id UUID гарантирует детерминированный порядок при
        одинаковом created_at (что реально при microsecond-точности Postgres).
        """
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
            .limit(limit)
        )
        result = await self._db.execute(stmt)
        rows = list(result.scalars().all())
        rows.reverse()
        return rows

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

    async def list_hypotheses_by_runs(
        self, run_ids: list[uuid.UUID]
    ) -> dict[uuid.UUID, list[Hypothesis]]:
        """Батч-запрос гипотез для нескольких run'ов одним SELECT.

        Решает N+1 в `SessionContext.get_best_strategy` / `get_untested_combos`
        (B11): раньше было до 21 запроса (1 для списка run'ов + 20 для
        гипотез каждого), теперь — 2 запроса (run'ы + батч гипотез).

        Returns:
            {run_id: [Hypothesis, ...]} — для run_id без гипотез пустой список.
        """
        if not run_ids:
            return {}
        stmt = (
            select(Hypothesis)
            .where(Hypothesis.run_id.in_(run_ids))
            .order_by(Hypothesis.dsr.desc().nullslast())
        )
        result = await self._db.execute(stmt)
        by_run: dict[uuid.UUID, list[Hypothesis]] = {rid: [] for rid in run_ids}
        for h in result.scalars().all():
            by_run.setdefault(h.run_id, []).append(h)
        return by_run

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

    # ── Session settings (encrypted credentials) ───────────────

    async def save_session_settings(
        self,
        session_id: str,
        llm_model: str,
        llm_api_key: str,
        openai_api_key: str,
        invest_token: str,
        invest_sandbox: bool = True,
    ) -> SessionSettings:
        """Создать или обновить настройки сессии.

        API-ключи шифруются через Fernet (см. `aqr.crypto`) перед записью.
        """
        await self.get_or_create_session(session_id)

        existing = await self._db.get(SessionSettings, session_id)
        if existing is None:
            settings = SessionSettings(
                session_id=session_id,
                llm_model=llm_model,
                llm_api_key_encrypted=encrypt_str(llm_api_key),
                openai_api_key_encrypted=encrypt_str(openai_api_key),
                invest_token_encrypted=encrypt_str(invest_token),
                invest_sandbox=invest_sandbox,
                updated_at=datetime.now(UTC),
            )
            self._db.add(settings)
        else:
            existing.llm_model = llm_model
            existing.llm_api_key_encrypted = encrypt_str(llm_api_key)
            existing.openai_api_key_encrypted = encrypt_str(openai_api_key)
            existing.invest_token_encrypted = encrypt_str(invest_token)
            existing.invest_sandbox = invest_sandbox
            existing.updated_at = datetime.now(UTC)
            settings = existing

        await self._db.flush()
        return settings

    async def load_session_settings(
        self, session_id: str
    ) -> SessionSettings | None:
        """Загрузить настройки или None если не заданы.

        Ключи НЕ расшифровываются на этом этапе (Fernet-токены остаются в БД).
        Используйте `decrypt_settings()` для получения plaintext.
        """
        return await self._db.get(SessionSettings, session_id)

    async def delete_session_settings(self, session_id: str) -> None:
        """Удалить настройки сессии (если были)."""
        existing = await self._db.get(SessionSettings, session_id)
        if existing is not None:
            await self._db.delete(existing)
            await self._db.flush()

    # ── Explore API ────────────────────────────────────────────

    async def list_all_hypotheses(
        self, session_id: str, limit: int = 100, offset: int = 0,
    ) -> list[dict]:
        """Все гипотезы с метриками для explore-таблицы."""
        stmt = (
            select(Hypothesis, Run)
            .join(Run, Hypothesis.run_id == Run.id)
            .where(Run.session_id == session_id)
            .order_by(Hypothesis.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        rows = await self._db.execute(stmt)
        result = []
        for hyp, run in rows.all():
            result.append({
                "id": str(hyp.id)[:8],
                "name": _hyp_name(hyp),
                "ticker": hyp.ticker,
                "family": hyp.family,
                "sharpe": hyp.sharpe,
                "dsr": hyp.dsr,
                "pbo": hyp.pbo,
                "verdict": _hyp_verdict(hyp),
                "owner": run.session_id,
                "updated": _ago(hyp.created_at),
                "run_id": str(run.id),
                "is_valid": hyp.is_valid,
            })
        return result

    async def get_hypothesis_detail(
        self, hyp_uuid: uuid.UUID, session_id: str,
    ) -> dict | None:
        """Детали гипотезы + run = для notebook."""
        stmt = (
            select(Hypothesis, Run)
            .join(Run)
            .where(Hypothesis.id == hyp_uuid, Run.session_id == session_id)
        )
        row = (await self._db.execute(stmt)).one_or_none()
        if row is None:
            return None
        hyp, run = row
        return {
            "id": str(hyp.id)[:8],
            "name": _hyp_name(hyp),
            "ticker": hyp.ticker,
            "family": hyp.family,
            "params": hyp.config_json,
            "sharpe": hyp.sharpe,
            "dsr": hyp.dsr,
            "pbo": hyp.pbo,
            "cpcv": hyp.cpcv,
            "max_drawdown": hyp.max_drawdown,
            "is_valid": hyp.is_valid,
            "verdict": _hyp_verdict(hyp),
            "goal": run.goal,
            "owner": run.session_id,
            "created_at": hyp.created_at.isoformat(),
            "updated_at": run.created_at.isoformat(),
        }

    async def get_explore_stats(self, session_id: str) -> dict:
        """Агрегатные метрики для статистики на explore."""
        from sqlalchemy import func
        hyp_count = (await self._db.execute(
            select(func.count(Hypothesis.id)).join(Run).where(Run.session_id == session_id)
        )).scalar() or 0
        n_valid = (await self._db.execute(
            select(func.count(Hypothesis.id))
            .join(Run)
            .where(Hypothesis.is_valid.is_(True), Run.session_id == session_id)
        )).scalar() or 0
        run_count = (await self._db.execute(
            select(func.count(Run.id)).where(Run.session_id == session_id)
        )).scalar() or 0
        return {
            "total_hypotheses": hyp_count,
            "total_runs": run_count,
            "approved": n_valid,
            "win_rate": round(n_valid / hyp_count * 100, 1) if hyp_count else 0,
        }

    async def get_recent_activity(self, session_id: str, days: int = 7) -> list[dict]:
        """События за N дней — создание гипотез и прогонов."""
        from datetime import timedelta
        cutoff = datetime.now(UTC) - timedelta(days=days)
        events: list[dict] = []

        # Новые гипотезы
        stmt_hyp = (
            select(Hypothesis, Run)
            .join(Run)
            .where(Hypothesis.created_at >= cutoff, Run.session_id == session_id)
            .order_by(Hypothesis.created_at.desc())
            .limit(50)
        )
        for hyp, run in (await self._db.execute(stmt_hyp)).all():
            events.append({
                "type": "create",
                "id": str(hyp.id)[:8],
                "name": _hyp_name(hyp),
                "actor": run.session_id,
                "ts": hyp.created_at.isoformat(),
                "ticker": hyp.ticker,
                "label": "hypothesis created",
            })

        # Новые прогоны
        stmt_run = (
            select(Run)
            .where(Run.created_at >= cutoff, Run.session_id == session_id)
            .order_by(Run.created_at.desc())
            .limit(20)
        )
        for run in (await self._db.execute(stmt_run)).scalars().all():
            events.append({
                "type": "rerun" if run.status == "done" else "created",
                "id": str(run.id)[:8],
                "name": run.goal[:60],
                "actor": run.session_id,
                "ts": run.created_at.isoformat(),
                "label": f"pipeline {run.status}",
            })

        events.sort(key=lambda e: e["ts"], reverse=True)
        return events[:50]


def _hyp_name(hyp: Hypothesis) -> str:
    return hyp.config_json.get("name", f"{hyp.family}/{hyp.ticker}")


def _hyp_verdict(hyp: Hypothesis) -> str:
    if hyp.is_valid:
        return "approved"
    if hyp.sharpe is None:
        return "idea"
    if hyp.sharpe > 1.0:
        return "screening"
    return "rejected"


def _ago(dt: datetime) -> str:
    """Human-friendly relative time."""
    from datetime import timedelta
    delta = datetime.now(UTC) - dt
    if delta < timedelta(hours=1):
        m = int(delta.total_seconds() / 60)
        return f"{m}m ago" if m else "just now"
    if delta < timedelta(days=1):
        return f"{int(delta.total_seconds() / 3600)}h ago"
    return f"{delta.days}d ago"


@dataclass(frozen=True)
class DecryptedSettings:
    """Plaintext-credentials, готовые к использованию в runtime.

    Создаётся через `RegistryStore.decrypt_settings()`.
    """

    session_id: str
    llm_model: str
    llm_api_key: str
    openai_api_key: str
    invest_token: str
    invest_sandbox: bool


def decrypt_settings(settings: SessionSettings) -> DecryptedSettings:
    """Расшифровать SessionSettings → plaintext. На ротации secret — RuntimeError."""
    return DecryptedSettings(
        session_id=settings.session_id,
        llm_model=settings.llm_model,
        llm_api_key=decrypt_str(settings.llm_api_key_encrypted),
        openai_api_key=decrypt_str(settings.openai_api_key_encrypted),
        invest_token=decrypt_str(settings.invest_token_encrypted),
        invest_sandbox=settings.invest_sandbox,
    )
