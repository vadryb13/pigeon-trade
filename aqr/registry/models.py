"""SQLAlchemy-модели: Run, Hypothesis, Session.

B17: каскадное удаление унифицировано на DB-level `ondelete="CASCADE"`
для всех FK (Session → Run → Hypothesis, Session → ChatMessage,
Session → SessionSettings). Это даёт атомарность на стороне Postgres
и не зависит от ORM `cascade="all, delete-orphan"` в коде приложения.
Менять только через миграцию Alembic (см. AGENTS.md инвариант 4).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Декларативная база с naming convention для Alembic autogenerate."""

    metadata_kwargs = {
        "naming_convention": {
            "ix": "ix_%(column_0_N_label)s",
            "uq": "uq_%(table_name)s_%(column_0_N_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    }


class Session(Base):
    """Сессия пользователя — контейнер для набора прогонов."""

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
    last_activity_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    runs: Mapped[list[Run]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    messages: Mapped[list[ChatMessage]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at",
    )


class ChatMessage(Base):
    """Одно сообщение в чате сессии (история диалога с агентом)."""

    __tablename__ = "chat_messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    meta: Mapped[dict | None] = mapped_column(
        "metadata", JSONB, nullable=True
    )  # node, tool_name, run_id и пр.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    __table_args__ = (
        Index("ix_chat_messages_session_created", "session_id", "created_at"),
    )

    session: Mapped[Session] = relationship(back_populates="messages")


class Run(Base):
    """Один прогон пайплайна: цель → N гипотез → результат."""

    __tablename__ = "runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    goal: Mapped[str] = mapped_column(Text, nullable=False)
    session_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="done")
    summary_metrics: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    session: Mapped[Session] = relationship(back_populates="runs")
    hypotheses: Mapped[list[Hypothesis]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class Hypothesis(Base):
    """Одна гипотеза в рамках прогона: семейство + тикер + параметры + метрики."""

    __tablename__ = "hypotheses"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    family: Mapped[str] = mapped_column(String(64), nullable=False)
    ticker: Mapped[str] = mapped_column(String(32), nullable=False)
    config_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1536), nullable=True)
    dsr: Mapped[float | None] = mapped_column(Float, nullable=True)
    pbo: Mapped[float | None] = mapped_column(Float, nullable=True)
    cpcv: Mapped[float | None] = mapped_column(Float, nullable=True)
    sharpe: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_drawdown: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_valid: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    run: Mapped[Run] = relationship(back_populates="hypotheses")

    # ivfflat-индекс добавляется отдельной миграцией после накопления данных
    # (ivfflat нельзя создавать на пустой таблице)
    __ivfflat_deferred__ = True


class SessionSettings(Base):
    """Per-session credentials: LLM model+key, OpenAI key (embeddings), Invest token.

    Хранятся зашифрованными (Fernet от AQR_SESSION_SECRET через HKDF).
    1:1 с sessions.id, FK CASCADE — при удалении сессии настройки удаляются.

    Заполняется через UI `POST /chat/{token}/settings`, читается при
    каждом WS-сообщении в `aqr.chat.ws._run_agent_for_session`.
    """

    __tablename__ = "session_settings"

    session_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        primary_key=True,
    )
    llm_model: Mapped[str] = mapped_column(String(120), nullable=False)
    llm_api_key_encrypted: Mapped[str] = mapped_column(Text, nullable=False)
    openai_api_key_encrypted: Mapped[str] = mapped_column(Text, nullable=False)
    invest_token_encrypted: Mapped[str] = mapped_column(Text, nullable=False)
    invest_sandbox: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )
