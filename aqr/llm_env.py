"""Утилита: временный override LLM/credentials env-переменных.

`litellm.completion()` / `litellm.acompletion()` и `openai.AsyncOpenAI()`
читают ключи из env (ANTHROPIC_API_KEY / OPENAI_API_KEY / GIGACHAT_CREDENTIALS).
Чтобы прокинуть per-session credentials без гонки между сессиями,
используется serialized override через `asyncio.Lock` (защита от
race condition B4 из docs/AUDIT.md).

litellm поддерживает `api_key=` per-call, но не все провайдеры это уважают.
Поэтому монкипатчим `os.environ` на время вызова под единым asyncio-локом —
так параллельные LLM-вызовы сериализуются на участке env-override, а
сам HTTP-запрос идёт асинхронно внутри блока.

NB: блокирует только другие LLM-вызовы, не event loop в целом.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aqr.registry import DecryptedSettings


_logger = logging.getLogger(__name__)

# Какие env-переменные пробрасываем
_LLM_ENV_KEYS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GIGACHAT_CREDENTIALS",
    "DEEPSEEK_API_KEY",
    "GEMINI_API_KEY",
)

# Ключи для fallback-поиска API-ключа из env
_API_KEY_ENV_KEYS = (
    "DEEPSEEK_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GIGACHAT_CREDENTIALS",
)

# Единый asyncio-лок для сериализации env-override при параллельных LLM-вызовах.
# Защищает от race в `finally` при concurrent `asyncio.gather` (B4).
_llm_env_lock = asyncio.Lock()


# Единый источник: model-substring → env-переменная.
# Порядок важен — первый match побеждает. Добавление нового провайдера —
# одна строка в этом списке, без изменения логики override.
_PROVIDER_ENV_MAP: list[tuple[str, str]] = [
    ("claude", "ANTHROPIC_API_KEY"),
    ("anthropic", "ANTHROPIC_API_KEY"),
    ("gigachat", "GIGACHAT_CREDENTIALS"),
    ("deepseek", "DEEPSEEK_API_KEY"),
    ("gemini", "GEMINI_API_KEY"),
]


def resolve_llm_credentials(model: str | None = None, caller: str = "") -> DecryptedSettings:
    """Получить credentials сессии или собрать из env (REST-путь).

    Пробует `current_credentials()` (per-session ContextVar), при None —
    собирает `DecryptedSettings` из env-переменных.
    Без модели + API-ключа — RuntimeError.

    Вызов safe вне WS-сессии (REST-ручки), т.к. импорт `current_credentials`
    ленивый внутри функции.
    """
    from aqr.graph.context import current_credentials
    from aqr.registry.store import DecryptedSettings

    creds = current_credentials()
    if creds is not None:
        return creds

    model = model or os.environ.get("AQR_LLM_MODEL", "")
    api_key = ""
    for k in _API_KEY_ENV_KEYS:
        api_key = os.environ.get(k, "")
        if api_key:
            break

    if not model or not api_key:
        raise RuntimeError(
            f"{caller or 'LLM call'}: no credentials available. "
            "Either configure via /chat/{token}/settings or set "
            "AQR_LLM_MODEL + one of DEEPSEEK_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY env vars."
        )
    return DecryptedSettings(
        session_id="rest",
        llm_model=model,
        llm_api_key=api_key,
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        invest_token=os.environ.get("INVEST_TOKEN", ""),
        invest_sandbox=os.environ.get("INVEST_SANDBOX", "1") != "0",
    )


def strip_markdown_fence(content: str) -> str:
    """Убрать markdown code fence (```json ... ```) из LLM-ответа.

    Возвращает содержимое без обрамляющих ```-строк.
    Если fence отсутствует — возвращает content как есть.
    Обрезает ```-строки только если они на отдельных строках (multiline fence).
    """
    content = content.strip()
    if not content.startswith("```"):
        return content
    lines = content.split("\n")
    if len(lines) >= 2 and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


@contextlib.contextmanager
def llm_env_override(creds: DecryptedSettings):
    """Контекст: на время вызова подменяет env credentials сессии.

    Безопасен для concurrent use благодаря `asyncio.Lock` — параллельные
    вызовы ждут друг друга на короткое время env-override/restore.
    Вызывающий код обязан `await` LLM-вызов ВНУТРИ блока, иначе восстановление
    env произойдёт до завершения HTTP-запроса.

    Провайдер определяется по model-substring через _PROVIDER_ENV_MAP.
    Fallback — OPENAI_API_KEY.
    """
    model = (creds.llm_model or "").lower()
    saved = {k: os.environ.get(k) for k in _LLM_ENV_KEYS}
    try:
        for k in _LLM_ENV_KEYS:
            os.environ.pop(k, None)
        env_var = "OPENAI_API_KEY"  # fallback
        for substr, var in _PROVIDER_ENV_MAP:
            if substr in model:
                env_var = var
                break
        os.environ[env_var] = creds.llm_api_key
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


async def acquire_llm_env_lock() -> contextlib.AbstractAsyncContextManager:
    """Async-обёртка над `llm_env_override` с serialization.

    Использование:
        async with await acquire_llm_env_lock() as sync:
            with sync(creds):
                resp = await litellm.acompletion(...)
    """
    return _AsyncEnvLockAdapter(_llm_env_lock)


class _AsyncEnvLockAdapter:
    """Async context manager, который после `acquire` возвращает синхронный
    `llm_env_override` для блока override.
    """

    def __init__(self, lock: asyncio.Lock) -> None:
        self._lock = lock

    async def __aenter__(self) -> _SyncEnvFactory:
        await self._lock.acquire()
        return _SyncEnvFactory(self._lock)

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self._lock.release()


class _SyncEnvFactory:
    """После захвата async-лока выдаёт синхронный ctx для env-override."""

    def __init__(self, lock: asyncio.Lock) -> None:
        self._lock = lock

    def __call__(self, creds: DecryptedSettings) -> contextlib.AbstractContextManager:
        return llm_env_override(creds)
