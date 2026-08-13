"""
Event bus — публикация событий во время исполнения пайплайна.

Каждый шаг (планирование, загрузка данных, генерация, бэктест, валидация)
пишет события в bus. UI (или CLI) их читает через SSE.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any


def _event_default(o: Any) -> Any:
    """Строгий JSON encoder для Event (B13).

    Поддерживает только известные безопасные типы. Любой другой объект
    (numpy scalar, custom dataclass без .to_dict, set, и т.п.) → TypeError.
    Раньше `default=str` молча сериализовал всё в repr, маскируя баги.
    """
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        # SSE-клиенты не любят NaN — превращаем в null с явным флагом.
        return None
    if isinstance(o, (str, int, bool, type(None), list, dict)):
        return o
    raise TypeError(
        f"Event JSON encoder: unsupported type {type(o).__name__}; "
        "convert explicitly before publishing."
    )


@dataclass
class Event:
    """Одно событие пайплайна."""

    run_id: str
    kind: str          # planning | data | generating | backtesting | validating | insight | done | error
    stage: str         # человекочитаемая стадия ("Загружаю SBER")
    message: str = ""  # подробность
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, default=_event_default)


class EventBus:
    """
    In-memory pub-sub с историей на run_id.

    Каждый run имеет:
    - список подписчиков (asyncio.Queue)
    - историю событий (для догоняющих подписчиков и финального отчёта)

    Concurrency: все мутации (`publish`, `subscribe`) защищены
    `_lock` (asyncio.Lock). Без лока два concurrent pipeline-run'а могут
    потерять подписчика или испортить `_done` (B6).
    """

    def __init__(self):
        self._history: dict[str, list[Event]] = {}
        self._subscribers: dict[str, list[asyncio.Queue]] = {}
        self._done: dict[str, asyncio.Event] = {}
        self._owners: dict[str, str] = {}
        self._finished_at: dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._history_limit = int(os.getenv("AQR_EVENT_HISTORY_LIMIT", "1000"))
        self._retention_seconds = float(os.getenv("AQR_EVENT_RETENTION_SECONDS", "3600"))

    async def register_run(self, run_id: str, session_id: str) -> None:
        """Register ownership before publishing; run IDs are never global data."""
        async with self._lock:
            self._history[run_id] = []
            self._subscribers[run_id] = []
            self._done[run_id] = asyncio.Event()
            self._owners[run_id] = session_id

    async def owns(self, run_id: str, session_id: str) -> bool:
        async with self._lock:
            return self._owners.get(run_id) == session_id

    def _prune_finished_locked(self, now: float) -> None:
        expired = [
            run_id for run_id, finished_at in self._finished_at.items()
            if now - finished_at >= self._retention_seconds
        ]
        for run_id in expired:
            self._history.pop(run_id, None)
            self._subscribers.pop(run_id, None)
            self._done.pop(run_id, None)
            self._owners.pop(run_id, None)
            self._finished_at.pop(run_id, None)

    async def publish(self, event: Event) -> None:
        # Fast-path без лока для чтения/апдейта dict.setdefault-append.
        # asyncio event loop — single-threaded, поэтому list.append атомарен.
        # Лок нужен ТОЛЬКО для атомарного "append + fan-out subscribers".
        # В однопоточном asyncio race возможен между проверкой длины
        # subscribers и put_nowait (если кто-то делает subscribe между
        # ними и вставит "тихий" хвост). Поэтому лочим.
        async with self._lock:
            self._prune_finished_locked(time.time())
            history = self._history.setdefault(event.run_id, [])
            history.append(event)
            if len(history) > self._history_limit:
                del history[:-self._history_limit]
            for q in self._subscribers.get(event.run_id, []):
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    logging.getLogger(__name__).warning(
                        "SSE queue full for run_id=%s, dropping event kind=%s",
                        event.run_id, event.kind,
                    )
            if event.kind in ("done", "error"):
                done = self._done.get(event.run_id)
                if done:
                    done.set()
                self._finished_at[event.run_id] = time.time()

    def history(self, run_id: str) -> list[Event]:
        # Read-only snapshot — атомарно под GIL.
        return list(self._history.get(run_id, []))

    async def subscribe(self, run_id: str) -> AsyncIterator[Event]:
        """SSE-подписка. Догоняет историю и стримит новое до события 'done'/'error'."""
        q: asyncio.Queue = asyncio.Queue(maxsize=1024)
        async with self._lock:
            self._subscribers.setdefault(run_id, []).append(q)
            history_snapshot = list(self._history.get(run_id, []))
            done = self._done.get(run_id)
        try:
            # 1. Догнать историю (снимок взят под локом)
            for ev in history_snapshot:
                yield ev
            if done and done.is_set():
                return
            # 2. Ждать новых
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=30.0)
                except TimeoutError:
                    # keep-alive tick — вернём "тишина", чтобы UI понимал что мы живы
                    continue
                yield ev
                if ev.kind in ("done", "error"):
                    return
        finally:
            async with self._lock:
                subs = self._subscribers.get(run_id, [])
                if q in subs:
                    subs.remove(q)


# Глобальная шина для процесса
BUS = EventBus()
