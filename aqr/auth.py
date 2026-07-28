"""Аутентификация сессий (SEC-1).

Подход: HMAC-подписанный session_id. Клиент получает подписанный токен при
создании сессии (например, через `POST /auth/session` или из query-param при
WebSocket-handshake). На `WS /chat/{token}` сервер проверяет подпись и
извлекает реальный session_id.

Если `AQR_SESSION_SECRET` не задан, генерируется ephemeral-ключ на старте
процесса — это OK для dev, но в проде клиенты теряют сессии при рестарте.

B14: `verify_token_async` дополнительно проверяет, что `session_id`
существует в БД (sessions table). Без этой проверки HMAC-валидный
токен продолжает работать после удаления сессии — клиент получит
404/500 на каждый запрос вместо явного close(1008).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
from typing import Final

_ALGO: Final = "sha256"
_SIGNATURE_PREFIX: Final = "v1."
_DEFAULT_TOKEN_TTL_SECONDS: Final = 60 * 60 * 24 * 30  # 30 дней

# Кешируется на уровне модуля, чтобы все вызовы в рамках процесса использовали
# один и тот же секрет (без env — ephemeral, генерируется один раз).
_EPHEMERAL_SECRET: bytes | None = None


def _get_secret() -> bytes:
    """Получить HMAC-секрет. Если env не задан — генерируется на процесс.

    Ephemeral-ключ означает: все ранее выданные токены становятся невалидны
    при рестарте процесса. В проде ОБЯЗАТЕЛЬНО задать AQR_SESSION_SECRET.
    """
    env = os.getenv("AQR_SESSION_SECRET")
    if env:
        return env.encode("utf-8")
    global _EPHEMERAL_SECRET
    if _EPHEMERAL_SECRET is None:
        _EPHEMERAL_SECRET = secrets.token_bytes(32)
    return _EPHEMERAL_SECRET


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode((s + pad).encode("ascii"))


def sign_session(session_id: str, ttl_seconds: int = _DEFAULT_TOKEN_TTL_SECONDS) -> str:
    """Подписать session_id, вернуть токен формата `v1.<b64id>.<b64sig>`.

    Args:
        session_id: произвольный ID сессии (любая ASCII-строка).
        ttl_seconds: сколько секунд токен валиден. По умолчанию 30 дней.

    Returns:
        токен, который клиент передаёт в `WS /chat/{token}` или в
        HTTP-заголовке `Authorization: Bearer <token>`.
    """
    import time

    payload = f"{int(time.time()) + ttl_seconds}:{session_id}".encode()
    sig = hmac.new(_get_secret(), payload, hashlib.sha256).digest()
    return f"{_SIGNATURE_PREFIX}{_b64url_encode(payload)}.{_b64url_encode(sig)}"


def verify_token(token: str) -> str | None:
    """Проверить токен (HMAC + TTL) и вернуть session_id, или None.

    Синхронный путь — НЕ проверяет существование session_id в БД.
    Используется в местах, где нет event loop (CLI, генерация токенов).

    Для WebSocket-handshake используй `verify_token_async` (B14).
    """
    import time

    if not token or not token.startswith(_SIGNATURE_PREFIX):
        return None
    body = token[len(_SIGNATURE_PREFIX):]
    parts = body.split(".", 1)
    if len(parts) != 2:
        return None
    payload_b64, sig_b64 = parts

    try:
        payload = _b64url_decode(payload_b64)
        sig = _b64url_decode(sig_b64)
    except Exception:
        return None

    expected_sig = hmac.new(_get_secret(), payload, hashlib.sha256).digest()
    if not hmac.compare_digest(sig, expected_sig):
        return None

    try:
        payload_str = payload.decode("utf-8")
        exp_str, _, session_id = payload_str.partition(":")
        if int(exp_str) < int(time.time()):
            return None
        return session_id
    except (ValueError, UnicodeDecodeError):
        return None


async def verify_token_async(token: str, db_session_factory=None) -> str | None:
    """Async-вариант `verify_token` с проверкой session_id в БД (B14).

    Если `db_session_factory=None` — пропускает DB-проверку (для тестов /
    случаев когда БД ещё не поднята). Иначе загружает Session по
    `session_id` и возвращает None, если сессия удалена.
    """
    from sqlalchemy import select

    session_id = verify_token(token)
    if session_id is None:
        return None
    if db_session_factory is None:
        return session_id

    try:
        async with db_session_factory() as db:
            from aqr.registry.models import Session
            stmt = select(Session.id).where(Session.id == session_id)
            row = (await db.execute(stmt)).first()
            return session_id if row is not None else None
    except Exception:
        # Если БД недоступна — fail open (HMAC-only). Это сознательный
        # выбор: недоступность БД не должна валить авторизацию.
        return session_id


def issue_default_token() -> str:
    """Создать токен для сессии 'default' — для dev/legacy-режима без auth."""
    return sign_session("default")
