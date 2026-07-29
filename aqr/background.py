"""Фоновая ретенция asyncio-задач (PERF-1).

`asyncio.create_task` возвращает задачу, на которую event loop держит только
слабую ссылку. Если функция-родитель завершилась и других strong refs нет,
задача может быть удалена GC до завершения. Это приводит к потере pipeline-run'ов
(«status=running forever») и несохранённым результатам при reload FastAPI.

Решение: держим strong-references в модульном set, удаляем через callback.
На shutdown ждём завершения всех задач (с таймаутом).
"""
from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import Coroutine
from typing import Any

# По умолчанию — до 64 одновременных pipeline-тасков; переопределяется через env.
_MAX_BACKGROUND_TASKS = int(os.getenv("AQR_MAX_BACKGROUND_TASKS", "64"))

# Strong-references на запущенные фоновые задачи (PERF-1).
_background_tasks: set[asyncio.Task] = set()


def schedule(coro: Coroutine[Any, Any, Any]) -> asyncio.Task:
    """Запустить корутину в фоне, удерживая strong-reference.

    Raises:
        RuntimeError: если превышен лимит одновременных задач
        (защита от утечки, когда клиент шлёт запросы быстрее, чем они завершаются).
    """
    if len(_background_tasks) >= _MAX_BACKGROUND_TASKS:
        raise RuntimeError(
            f"Превышен лимит фоновых задач ({_MAX_BACKGROUND_TASKS}). "
            "Смотрите AQR_MAX_BACKGROUND_TASKS или уменьшите нагрузку."
        )
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


async def drain(timeout: float = 30.0) -> None:
    """Дождаться завершения всех фоновых задач с таймаутом.

    Вызывается из FastAPI lifespan при shutdown.
    """
    if not _background_tasks:
        return
    # Задачи продолжат работать пока жив event loop. Best-effort ждём.
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(
            asyncio.gather(*_background_tasks, return_exceptions=True),
            timeout=timeout,
        )


def active_count() -> int:
    """Текущее количество активных фоновых задач (для мониторинга)."""
    return len(_background_tasks)
