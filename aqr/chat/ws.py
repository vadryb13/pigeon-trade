"""WebSocket chat: двусторонний диалог с агентом.

Контракт JSON-сообщений от клиента:
    {"type": "message", "content": "проверь momentum на Сбере"}
    {"type": "resume"}                                    -- запрос истории
    {"type": "ping"}                                      -- keepalive

Контракт от сервера:
    {"type": "history",    "messages": [{role, content, ts}]}
    {"type": "user_echo",  "content": "..."}              -- подтверждение user msg
    {"type": "progress",   "node": "...", "data": {...}}  -- из графа
    {"type": "tool_call",  "name": "...", "args": {...}}
    {"type": "tool_result","name": "...", "result": {...}}
    {"type": "assistant",  "content": "..."}              -- финальный текст
    {"type": "done",       "narrative": "...", "run_id": "..."}
    {"type": "error",      "message": "..."}
    {"type": "pong"}

Auth: HMAC-подпись session_id обязательна. Per-session credentials
(LLM/OpenAI/Invest keys) загружаются из session_settings на handshake
и пробрасываются в графа через ContextVar (см. aqr.agent.context).
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from aqr.agent.context import reset_credentials, set_credentials
from aqr.agent.graph import _msg_content, _msg_role, get_agent
from aqr.auth import verify_token
from aqr.crypto import decrypt_str
from aqr.db import _async_session_factory
from aqr.registry import DecryptedSettings, RegistryStore, SessionSettings

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Helpers ──────────────────────────────────────────────────────

async def _save_history(
    session_id: str,
    role: str,
    content: str,
    meta: dict | None = None,
) -> None:
    """Сохранить сообщение в БД. Тихий отказ при ошибке (graceful)."""
    try:
        async with _async_session_factory() as db:
            store = RegistryStore(db)
            await store.save_chat_message(
                session_id=session_id, role=role, content=content, meta=meta,
            )
            await db.commit()
    except Exception:
        logger.warning("Не удалось сохранить сообщение", exc_info=True)


async def _load_history(session_id: str, limit: int = 200) -> list[dict[str, Any]]:
    """Загрузить последние N сообщений сессии. Пустой список при ошибке."""
    try:
        async with _async_session_factory() as db:
            store = RegistryStore(db)
            msgs = await store.list_chat_history(session_id, limit=limit)
            return [
                {
                    "id": str(m.id),
                    "role": m.role,
                    "content": m.content,
                    "created_at": m.created_at.isoformat(),
                }
                for m in msgs
            ]
    except Exception:
        logger.warning("Не удалось загрузить историю", exc_info=True)
        return []


async def _load_credentials(session_id: str) -> DecryptedSettings | None:
    """Загрузить расшифрованные credentials сессии или None.

    Бросает исключение при ошибке БД — credentials критичны для
    безопасности (без них LLM/Invest не работают, и мы не хотим
    тихо игнорировать пропавший доступ к данным).
    """
    async with _async_session_factory() as db:
        store = RegistryStore(db)
        settings = await store.load_session_settings(session_id)
    if settings is None:
        return None
    return DecryptedSettings(
        session_id=settings.session_id,
        llm_model=settings.llm_model,
        llm_api_key=decrypt_str(settings.llm_api_key_encrypted),
        openai_api_key=decrypt_str(settings.openai_api_key_encrypted),
        invest_token=decrypt_str(settings.invest_token_encrypted),
        invest_sandbox=settings.invest_sandbox,
    )


async def _send_json(ws: WebSocket, payload: dict[str, Any]) -> None:
    """Отправить JSON клиенту (обёртка для удобства)."""
    await ws.send_text(json.dumps(payload, ensure_ascii=False))


# ── WS endpoint ──────────────────────────────────────────────────

@router.websocket("/chat/{token}")
async def chat_ws(websocket: WebSocket, token: str):
    """WebSocket: двусторонний диалог пользователя с агентом.

    Path `token` — это HMAC-подписанный session_id (см. `aqr.auth.sign_session`).
    Handshake проверяет подпись; невалидный токен → close(1008).

    Per-session credentials загружаются из session_settings на handshake
    и пробрасываются в графа через ContextVar. Без настроенных credentials
    сообщения `{type: "message"}` отвергаются с ошибкой + ссылкой на
    /chat/{token}/settings.

    Принимает:
        {type: "message", content: "..."} — запуск графа агента
        {type: "resume"} — отдать историю
        {type: "ping"} — keepalive
    """
    session_id = verify_token(token)
    if session_id is None:
        await websocket.close(code=1008, reason="invalid token")
        return

    credentials = await _load_credentials(session_id)

    await websocket.accept()

    await _send_json(websocket, {
        "type": "connected",
        "session_id": session_id,
        "credentials_configured": credentials is not None,
    })
    if not credentials:
        await _send_json(websocket, {
            "type": "error",
            "message": (
                "Session settings not configured. "
                f"Open /chat/{token}/settings and enter credentials."
            ),
        })
    history = await _load_history(session_id)
    if history:
        await _send_json(websocket, {"type": "history", "messages": history})

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await _send_json(websocket, {
                    "type": "error", "message": "Invalid JSON",
                })
                continue

            msg_type = payload.get("type")

            if msg_type == "ping":
                await _send_json(websocket, {"type": "pong"})

            elif msg_type == "resume":
                history = await _load_history(session_id)
                await _send_json(websocket, {"type": "history", "messages": history})

            elif msg_type == "message":
                content = (payload.get("content") or "").strip()
                if not content:
                    continue

                if credentials is None:
                    # Reload — может пользователь только что настроил
                    credentials = await _load_credentials(session_id)
                    if credentials is None:
                        await _send_json(websocket, {
                            "type": "error",
                            "message": (
                                "Session settings not configured. "
                                f"Open /chat/{token}/settings and enter credentials."
                            ),
                        })
                        continue

                await _save_history(session_id, "user", content)
                await _send_json(websocket, {
                    "type": "user_echo", "content": content,
                })

                try:
                    await _run_agent_for_session(
                        websocket=websocket,
                        session_id=session_id,
                        message=content,
                        credentials=credentials,
                    )
                except Exception as e:
                    logger.exception("Agent crashed in WS")
                    await _send_json(websocket, {
                        "type": "error", "message": f"Agent error: {e}",
                    })

            else:
                await _send_json(websocket, {
                    "type": "error",
                    "message": f"Unknown message type: {msg_type!r}",
                })

    except WebSocketDisconnect:
        logger.info("WS disconnected: session_id=%s", session_id)
    except Exception as e:
        logger.exception("WS error")
        with contextlib.suppress(Exception):
            await _send_json(websocket, {"type": "error", "message": str(e)})


# ── Agent runner (стримит события в WS) ─────────────────────────

async def _run_agent_for_session(
    websocket: WebSocket,
    session_id: str,
    message: str,
    credentials: DecryptedSettings,
) -> None:
    """Запустить графа агента и стримить узлы/результаты в WS.

    Credentials пробрасываются через ContextVar на время работы графа —
    planner/narrator/reviewer/Embedder/TInvestAdapter читают их
    через `current_credentials()`.
    """
    await _send_json(websocket, {"type": "progress", "node": "start", "data": {"goal": message}})

    cred_token = set_credentials(credentials)
    try:
        result = await _run_agent_inner(websocket, session_id, message)
    finally:
        reset_credentials(cred_token)
    return result  # noqa: F706 — explicit for type checker


async def _run_agent_inner(
    websocket: WebSocket,
    session_id: str,
    message: str,
) -> None:
    """Тело runner'а — обёрнуто в ContextVar lifecycle в _run_agent_for_session."""
    t0 = time.time()
    initial_state: dict[str, Any] = {
        "messages": [{"role": "user", "content": message}],
        "session_id": session_id,
        "goal": message,
        "plan": None,
        "prices": None,
        "hypotheses": None,
        "results": None,
        "pbo": None,
        "insights": None,
        "narrative": None,
        "session_context_prompt": "",
        "step": "",
        "error": None,
        "elapsed_seconds": 0.0,
        "n_tested": 0,
        "n_survived": 0,
    }

    try:
        agent = get_agent()
        try:
            async for ev in agent.astream(initial_state, stream_mode="values"):
                if not isinstance(ev, dict):
                    continue
                node = ev.get("step", "")
                if node:
                    await _send_json(websocket, {
                        "type": "progress",
                        "node": node,
                        "data": _state_summary(ev),
                    })
                last_state = ev
        except Exception:
            last_state = await agent.ainvoke(initial_state)
            await _send_json(websocket, {"type": "progress", "node": "final", "data": _state_summary(last_state)})

        final_messages = last_state.get("messages", []) if isinstance(last_state, dict) else []
        assistant_text = ""
        for m in reversed(final_messages):
            if _msg_role(m) == "assistant":
                assistant_text = _msg_content(m)
                break

        narrative = last_state.get("narrative") if isinstance(last_state, dict) else None
        plan = last_state.get("plan") if isinstance(last_state, dict) else None
        results = last_state.get("results") if isinstance(last_state, dict) else None

        if assistant_text:
            await _save_history(
                session_id, "assistant", assistant_text,
                meta={
                    "narrative": (narrative or "")[:500] if narrative else "",
                    "n_results": len(results) if results else 0,
                },
            )

        await _send_json(websocket, {
            "type": "done",
            "narrative": narrative or assistant_text,
            "assistant": assistant_text,
            "elapsed_seconds": round(time.time() - t0, 2),
            "n_results": len(results) if results else 0,
            "plan_tickers": plan.get("tickers") if isinstance(plan, dict) else None,
        })

    except Exception as e:
        await _send_json(websocket, {
            "type": "error",
            "message": f"Agent failed: {e}",
        })


def _state_summary(state: dict[str, Any]) -> dict[str, Any]:
    """Краткая сводка по state для progress-сообщения (без тяжёлых полей)."""
    return {
        "step": state.get("step"),
        "n_tested": state.get("n_tested", 0),
        "n_survived": state.get("n_survived", 0),
        "has_plan": state.get("plan") is not None,
        "has_results": state.get("results") is not None,
        "has_pbo": state.get("pbo") is not None,
        "has_narrative": state.get("narrative") is not None,
    }
