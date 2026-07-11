"""Database session factory and FastAPI dependency."""

from __future__ import annotations

import os
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:aqr@localhost:5432/aqr",
)

_engine = create_async_engine(DB_URL, echo=False)
_async_session_factory = async_sessionmaker(_engine, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI-зависимость: yield асинхронной сессии БД."""
    async with _async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
