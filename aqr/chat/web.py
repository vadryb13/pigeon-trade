"""Web UI: HTML-страницы чата и настроек, endpoint для токенов и settings.

Endpoints:
- GET  /chat                          — HTML-страница чата
- GET  /chat/new?session_id=...       — HMAC-токен для новой сессии
- GET  /chat/{token}/settings         — HTML-форма настроек (credentials)
- POST /chat/{token}/settings         — сохранение credentials в session_settings
- GET  /chat/{token}/settings/status  — JSON {configured: bool, llm_model?: str}

Все settings-endpoint'ы требуют HMAC-валидный токен (как /chat/{token} WS).
Логика чата — в `aqr.chat.ws`. Этот модуль — только статика + токены.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from aqr.auth import sign_session, verify_token
from aqr.db import _async_session_factory
from aqr.registry import RegistryStore

_CHAT_TEMPLATE = Path(__file__).parent / "templates" / "chat.html"
_SETTINGS_TEMPLATE = Path(__file__).parent / "templates" / "settings.html"

router = APIRouter()


# ── Helpers ──────────────────────────────────────────────────────


def _resolve_session_id(token: str) -> str:
    """HMAC-валидация токена → session_id, иначе 403."""
    sid = verify_token(token)
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
    session_id = _resolve_session_id(token)
    if not _SETTINGS_TEMPLATE.exists():
        raise HTTPException(
            status_code=500,
            detail=f"settings template not found: {_SETTINGS_TEMPLATE}",
        )

    async with _async_session_factory() as db:
        existing = await RegistryStore(db).load_session_settings(session_id)

    if existing is not None:
        # Уже настроено → не показываем форму, редиректим в чат
        return RedirectResponse(url=f"/chat/{token}", status_code=303)

    return HTMLResponse(content=_SETTINGS_TEMPLATE.read_text(encoding="utf-8"))


@router.post("/chat/{token}/settings")
async def settings_save(
    token: str,
    llm_model: str = Form(..., min_length=1, max_length=120),
    llm_api_key: str = Form(..., min_length=1, max_length=512),
    openai_api_key: str = Form(..., min_length=1, max_length=512),
    invest_token: str = Form(..., min_length=1, max_length=512),
    invest_sandbox: str = Form("off"),
) -> RedirectResponse:
    """Сохранить credentials в session_settings, редирект на чат."""
    session_id = _resolve_session_id(token)
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

    return RedirectResponse(url=f"/chat/{token}", status_code=303)


@router.get("/chat/{token}/settings/status")
async def settings_status(token: str) -> JSONResponse:
    """JSON для UI: настроено или нет, и какая модель."""
    session_id = _resolve_session_id(token)

    async with _async_session_factory() as db:
        existing = await RegistryStore(db).load_session_settings(session_id)

    if existing is None:
        return JSONResponse({"configured": False})
    return JSONResponse(
        {"configured": True, "llm_model": existing.llm_model}
    )


__all__ = ["router", "SettingsPayload"]
