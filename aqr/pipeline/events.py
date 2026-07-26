"""
Event bus — публикация событий во время исполнения пайплайна.

Каждый шаг (планирование, загрузка данных, генерация, бэктест, валидация)
пишет события в bus. UI (или CLI) их читает через SSE.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Event:
    """Одно событие пайплайна."""

    run_id: str
    kind: str          # planning | data | generating | backtesting | validating | insight | done | error
    stage: str         # человекочитаемая стадия ("Загружаю MOEX SBER")
    message: str = ""  # подробность
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, default=str)


class EventBus:
    """
    In-memory pub-sub с историей на run_id.

    Каждый run имеет:
    - список подписчиков (asyncio.Queue)
    - историю событий (для догоняющих подписчиков и финального отчёта)
    """

    def __init__(self):
        self._history: dict[str, list[Event]] = {}
        self._subscribers: dict[str, list[asyncio.Queue]] = {}
        self._done: dict[str, asyncio.Event] = {}

    def new_run(self) -> str:
        run_id = str(uuid.uuid4())
        self._history[run_id] = []
        self._subscribers[run_id] = []
        self._done[run_id] = asyncio.Event()
        return run_id

    async def publish(self, event: Event) -> None:
        self._history.setdefault(event.run_id, []).append(event)
        for q in self._subscribers.get(event.run_id, []):
            with contextlib.suppress(asyncio.QueueFull):
                q.put_nowait(event)
        if event.kind in ("done", "error"):
            done = self._done.get(event.run_id)
            if done:
                done.set()

    def history(self, run_id: str) -> list[Event]:
        return list(self._history.get(run_id, []))

    async def subscribe(self, run_id: str) -> AsyncIterator[Event]:
        """SSE-подписка. Догоняет историю и стримит новое до события 'done'/'error'."""
        q: asyncio.Queue = asyncio.Queue(maxsize=1024)
        self._subscribers.setdefault(run_id, []).append(q)
        try:
            # 1. Догнать историю
            for ev in list(self._history.get(run_id, [])):
                yield ev
            done = self._done.get(run_id)
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
            subs = self._subscribers.get(run_id, [])
            if q in subs:
                subs.remove(q)


# Глобальная шина для процесса
BUS = EventBus()
