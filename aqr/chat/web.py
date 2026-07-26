"""Web UI: HTML-страница чата и endpoint для генерации токена.

Бэкенд отдаёт статическую HTML-страницу (`GET /chat`) и endpoint для
получения HMAC-подписанного токена (`GET /chat/new?session_id=...`).

Сама логика чата работает через WebSocket (`/chat/{token}`), который
уже реализован в `aqr.chat.ws`. Этот модуль только отдаёт статику
и выпускает токены — никакой бизнес-логики пайплайна здесь нет.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse

from aqr.auth import sign_session

_TEMPLATE_PATH = Path(__file__).parent / "templates" / "chat.html"


router = APIRouter()


@router.get("/chat", response_class=HTMLResponse)
async def chat_page() -> HTMLResponse:
    """Отдать HTML-страницу чата (SPA с inline JS + WebSocket-клиентом)."""
    if not _TEMPLATE_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"chat template not found: {_TEMPLATE_PATH}",
        )
    html = _TEMPLATE_PATH.read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@router.get("/chat/new")
async def chat_new(
    session_id: str = Query(..., min_length=1, max_length=64),
) -> JSONResponse:
    """Выпустить HMAC-подписанный токен для новой сессии.

    Если `AQR_REQUIRE_WS_AUTH=0` (dev/legacy режим), токен не нужен —
    возвращаем `token=null`. Клиент подставляет session_id в URL WS
    напрямую, без подписи.
    """
    require_auth = os.getenv("AQR_REQUIRE_WS_AUTH", "1") != "0"
    if require_auth:
        token = sign_session(session_id)
    else:
        token = None
    return JSONResponse({"session_id": session_id, "token": token})


__all__ = ["router"]
