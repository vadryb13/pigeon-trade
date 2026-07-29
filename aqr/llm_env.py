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
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aqr.registry import DecryptedSettings


# Какие env-переменные пробрасываем
_LLM_ENV_KEYS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GIGACHAT_CREDENTIALS",
    "DEEPSEEK_API_KEY",
    "GEMINI_API_KEY",
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
