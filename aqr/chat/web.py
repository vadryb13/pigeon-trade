"""Web UI: HTML-страницы чата и настроек, endpoint для токенов и settings.

Endpoints:
- GET  /chat                          — HTML-страница чата
- GET  /chat/new?session_id=...       — HMAC-токен для новой сессии
- GET  /chat/{token}/settings         — HTML-форма настроек (credentials)
- POST /chat/{token}/settings         — сохранение credentials в session_settings
- GET  /chat/{token}/settings/status  — JSON {configured: bool, llm_model?: str}

Все settings-endpoint'ы требуют HMAC-валидный токен (как /chat/{token} WS).
Логика чата — в `aqr.chat.ws`. Этот модуль — только статика + токены.

B15: rate limit на POST /settings (token bucket per-IP) — защищает от
credential-overwrite атаки с подобранным/украденным токеном.
"""
from __future__ import annotations

import time
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from aqr.auth import sign_session, verify_token, verify_token_async
from aqr.db import _async_session_factory
from aqr.registry import RegistryStore

_CHAT_TEMPLATE = Path(__file__).parent / "templates" / "chat.html"
_SETTINGS_TEMPLATE = Path(__file__).parent / "templates" / "settings.html"

router = APIRouter()


# ── Rate limiting (B15) ─────────────────────────────────────────
#
# Простой in-memory token bucket per-IP для POST /settings.
# Не персистится между рестартами — намеренно: атакующий тоже не получает
# кросс-процессное преимущество. В проде заменить на slowapi/Redis.

_RATE_BUCKET_CAPACITY = 5  # запросов
_RATE_BUCKET_REFILL_PER_SEC = 1.0 / 30.0  # 1 токен каждые 30 сек
_rate_buckets: dict[str, tuple[float, float]] = {}  # ip → (tokens, last_ts)


def _prune_rate_buckets(max_age: float = 300.0) -> None:
    """Удалить записи старше max_age секунд чтобы избежать утечки памяти."""
    now = time.monotonic()
    stale = [ip for ip, (_, last_ts) in _rate_buckets.items() if now - last_ts > max_age]
    for ip in stale:
        del _rate_buckets[ip]


def _rate_limit_consume(ip: str) -> bool:
    """Возвращает True если запрос разрешён, False если rate-limit."""
    now = time.monotonic()
    tokens, last_ts = _rate_buckets.get(ip, (_RATE_BUCKET_CAPACITY, now))
    elapsed = now - last_ts
    tokens = min(
        _RATE_BUCKET_CAPACITY,
        tokens + elapsed * _RATE_BUCKET_REFILL_PER_SEC,
    )
    if tokens < 1.0:
        _rate_buckets[ip] = (tokens, now)
        return False
    _rate_buckets[ip] = (tokens - 1.0, now)
    _prune_rate_buckets()
    return True


def _client_ip(request: Request) -> str:
    """Извлечь IP клиента, учитывая X-Forwarded-For если есть."""
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# ── Helpers ──────────────────────────────────────────────────────


async def _resolve_session_id(token: str) -> str:
    """HMAC-валидация токена + проверка session_id в БД (B14). Иначе 403."""
    sid = await verify_token_async(token, _async_session_factory)
    if sid is None:
        raise HTTPException(status_code=403, detail="invalid token")
    return sid


# ── /chat (HTML) ────────────────────────────────────────────────


@router.get("/chat", response_class=HTMLResponse)
async def chat_page() -> HTMLResponse:
    """Отдать HTML-страницу чата (SPA с inline JS + WebSocket-клиентом)."""
    if not _CHAT_TEMPLATE.exists():
        raise HTTPException(
            status_code=500,
            detail=f"chat template not found: {_CHAT_TEMPLATE}",
        )
    return HTMLResponse(content=_CHAT_TEMPLATE.read_text(encoding="utf-8"))


# ── /chat/new (token issuance) ──────────────────────────────────


@router.get("/chat/new")
async def chat_new(
    session_id: str = Query(..., min_length=1, max_length=64),
) -> JSONResponse:
    """Выпустить HMAC-токен для новой сессии."""
    token = sign_session(session_id)
    return JSONResponse({"session_id": session_id, "token": token})


# ── /chat/{token}/settings (form + save) ────────────────────────


class SettingsPayload(BaseModel):
    """POST body для /chat/{token}/settings."""

    llm_model: str = Field(..., min_length=1, max_length=120)
    llm_api_key: str = Field(..., min_length=1, max_length=512)
    openai_api_key: str = Field(..., min_length=1, max_length=512)
    invest_token: str = Field(..., min_length=1, max_length=512)
    invest_sandbox: bool = True


@router.get("/chat/{token}/settings", response_class=HTMLResponse)
async def settings_page(token: str) -> HTMLResponse:
    """HTML-форма для ввода credentials. Если уже настроено — редирект на чат."""
    session_id = await _resolve_session_id(token)
    if not _SETTINGS_TEMPLATE.exists():
        raise HTTPException(
            status_code=500,
            detail=f"settings template not found: {_SETTINGS_TEMPLATE}",
        )

    async with _async_session_factory() as db:
        existing = await RegistryStore(db).load_session_settings(session_id)

    if existing is not None:
        return RedirectResponse(url="/chat", status_code=303)

    return HTMLResponse(content=_SETTINGS_TEMPLATE.read_text(encoding="utf-8"))


@router.post("/chat/{token}/settings")
async def settings_save(
    request: Request,
    token: str,
    llm_model: str = Form(..., min_length=1, max_length=120),
    llm_api_key: str = Form(..., min_length=1, max_length=512),
    openai_api_key: str = Form(..., min_length=1, max_length=512),
    invest_token: str = Form(..., min_length=1, max_length=512),
    invest_sandbox: str = Form("off"),
) -> RedirectResponse:
    """Сохранить credentials в session_settings, редирект на чат."""
    # B15: rate limit перед auth — иначе можно перебирать токены.
    if not _rate_limit_consume(_client_ip(request)):
        raise HTTPException(status_code=429, detail="rate limit exceeded")

    session_id = await _resolve_session_id(token)
    sandbox = invest_sandbox == "on"

    async with _async_session_factory() as db:
        store = RegistryStore(db)
        await store.save_session_settings(
            session_id=session_id,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            openai_api_key=openai_api_key,
            invest_token=invest_token,
            invest_sandbox=sandbox,
        )
        await db.commit()

    return RedirectResponse(url="/chat", status_code=303)


@router.get("/chat/{token}/settings/status")
async def settings_status(token: str) -> JSONResponse:
    """JSON для UI: настроено или нет, и какая модель."""
    session_id = await _resolve_session_id(token)

    async with _async_session_factory() as db:
        existing = await RegistryStore(db).load_session_settings(session_id)

    if existing is None:
        return JSONResponse({"configured": False})
    return JSONResponse(
        {"configured": True, "llm_model": existing.llm_model}
    )


__all__ = ["router", "SettingsPayload"]
