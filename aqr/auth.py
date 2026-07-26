"""Аутентификация сессий (SEC-1).

Подход: HMAC-подписанный session_id. Клиент получает подписанный токен при
создании сессии (например, через `POST /auth/session` или из query-param при
WebSocket-handshake). На `WS /chat/{token}` сервер проверяет подпись и
извлекает реальный session_id.

Если `AQR_SESSION_SECRET` не задан, генерируется ephemeral-ключ на старте
процесса — это OK для dev, но в проде клиенты теряют сессии при рестарте.
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
    """Проверить токен и вернуть session_id, или None если невалидный/просрочен.

    Не бросает исключения — при любой ошибке возвращает None. Это позволяет
    WebSocket-handshake всегда отвечать close(1008), а не 500.
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


def issue_default_token() -> str:
    """Создать токен для сессии 'default' — для dev/legacy-режима без auth."""
    return sign_session("default")
