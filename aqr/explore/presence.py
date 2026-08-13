"""Presence tracking and pessimistic lock for Explore UI.

In-memory tracker with asyncio-based cleanup. Locks auto-release after 5 min.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncIterator

_logger = logging.getLogger(__name__)

_LOCK_TTL = 300  # 5 min auto-release
_HEARTBEAT_TTL = 60  # 60s without heartbeat → offline
_CLEANUP_INTERVAL = 30  # check every 30s


class PresenceTracker:
    """Track active sessions and hypothesis locks.

    Thread-safe (asyncio.Lock). Publishes state changes to SSE subscribers.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        # session_id → {"name": str, "hyp_id": str|None, "last_seen": float}
        self._sessions: dict[str, dict] = {}
        # hyp_id → {"session_id": str, "locked_at": float}
        self._locks: dict[str, dict] = {}
        # SSE subscribers → asyncio.Queue[dict]
        self._subscribers: list[asyncio.Queue] = []
        self._cleanup_task: asyncio.Task | None = None

    # ── Session management ──────────────────────────────────────

    async def heartbeat(self, session_id: str, name: str, hyp_id: str | None = None) -> None:
        async with self._lock:
            self._sessions[session_id] = {"name": name, "hyp_id": hyp_id, "last_seen": time.monotonic()}
            # Extend lock if held by this session
            if hyp_id and hyp_id in self._locks and self._locks[hyp_id]["session_id"] == session_id:
                self._locks[hyp_id]["locked_at"] = time.monotonic()
        await self._broadcast({"type": "presence", "sessions": self._snapshot()})

    async def remove(self, session_id: str) -> None:
        async with self._lock:
            self._sessions.pop(session_id, None)
            for hid, lock in list(self._locks.items()):
                if lock["session_id"] == session_id:
                    del self._locks[hid]
                    await self._broadcast_locked({"type": "unlock", "hyp_id": hid})
        await self._broadcast({"type": "presence", "sessions": self._snapshot()})

    def _snapshot(self) -> list[dict]:
        now = time.monotonic()
        return [
            {"name": s["name"], "hyp_id": s["hyp_id"]}
            for s in self._sessions.values()
            if now - s["last_seen"] < _HEARTBEAT_TTL
        ]

    # ── Lock management ─────────────────────────────────────────

    async def lock(self, hyp_id: str, session_id: str, name: str) -> dict:
        async with self._lock:
            existing = self._locks.get(hyp_id)
            if existing and existing["session_id"] != session_id:
                return {"ok": False, "locked_by": existing["session_id"]}
            self._locks[hyp_id] = {"session_id": session_id, "locked_at": time.monotonic()}
            self._sessions.setdefault(session_id, {"name": name, "hyp_id": hyp_id, "last_seen": time.monotonic()})
        await self._broadcast_locked({"type": "lock", "hyp_id": hyp_id, "session_id": session_id, "name": name})
        return {"ok": True}

    async def unlock(self, hyp_id: str, session_id: str) -> dict:
        async with self._lock:
            if self._locks.get(hyp_id, {}).get("session_id") != session_id:
                return {"ok": False, "error": "not your lock"}
            del self._locks[hyp_id]
        await self._broadcast_locked({"type": "unlock", "hyp_id": hyp_id})
        return {"ok": True}

    def _get_locked_hyp_ids(self) -> set[str]:
        now = time.monotonic()
        stale = [hid for hid, lk in self._locks.items() if now - lk["locked_at"] > _LOCK_TTL]
        for hid in stale:
            del self._locks[hid]
        return set(self._locks.keys())

    # ── SSE subscribers ─────────────────────────────────────────

    async def subscribe(self) -> AsyncIterator[str]:
        queue: asyncio.Queue = asyncio.Queue()
        async with self._lock:
            self._subscribers.append(queue)
        try:
            # Send initial state
            yield self._encode({"type": "presence", "sessions": self._snapshot()})
            locked = [hid for hid in self._locks]
            if locked:
                yield self._encode({"type": "locked_ids", "ids": locked})
            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield self._encode(msg)
                except TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            async with self._lock:
                if queue in self._subscribers:
                    self._subscribers.remove(queue)

    async def _broadcast(self, msg: dict) -> None:
        async with self._lock:
            qs = list(self._subscribers)
        for q in qs:
            with contextlib.suppress(asyncio.QueueFull):
                q.put_nowait(msg)

    async def _broadcast_locked(self, msg: dict) -> None:
        """Broadcast to subscribers excluding the sender."""
        async with self._lock:
            qs = list(self._subscribers)
        for q in qs:
            with contextlib.suppress(asyncio.QueueFull):
                q.put_nowait(msg)

    @staticmethod
    def _encode(msg: dict) -> str:
        return f"data: {json.dumps(msg, default=str)}\n\n"

    # ── Cleanup ─────────────────────────────────────────────────

    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(_CLEANUP_INTERVAL)
            now = time.monotonic()
            async with self._lock:
                # Remove stale sessions
                stale_sessions = [
                    sid for sid, s in self._sessions.items()
                    if now - s["last_seen"] > _HEARTBEAT_TTL
                ]
                for sid in stale_sessions:
                    self._sessions.pop(sid, None)
                # Auto-release stale locks
                stale_locks = [
                    hid for hid, lk in self._locks.items()
                    if now - lk["locked_at"] > _LOCK_TTL
                ]
                for hid in stale_locks:
                    self._locks.pop(hid, None)
            if stale_sessions or stale_locks:
                await self._broadcast({"type": "presence", "sessions": self._snapshot()})
                for hid in stale_locks:
                    await self._broadcast({"type": "unlock", "hyp_id": hid})

    def start(self) -> None:
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())


TRACKER = PresenceTracker()
