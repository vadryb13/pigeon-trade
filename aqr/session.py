"""Database session factory and FastAPI dependency."""

from __future__ import annotations

import os
import threading
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:aqr@localhost:5432/aqr",
)


class _LazySessionFactory:
    """Lazy-init async_sessionmaker. Engine создаётся при первом вызове,
    а не на import — чтобы тесты без БД могли импортировать модуль."""

    def __init__(self) -> None:
        self._factory: async_sessionmaker[AsyncSession] | None = None
        self._lock = threading.Lock()

    def _init(self) -> async_sessionmaker[AsyncSession]:
        if self._factory is not None:
            return self._factory
        with self._lock:
            if self._factory is None:
                engine = create_async_engine(DB_URL, echo=False)
                self._factory = async_sessionmaker(engine, expire_on_commit=False)
            return self._factory

    def __call__(self, **kwargs):
        return self._init().__call__(**kwargs)


async_session_factory = _LazySessionFactory()


async def get_db() -> AsyncGenerator[AsyncSession]:
    """FastAPI-зависимость: yield асинхронной сессии БД."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
