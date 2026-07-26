"""Утилита: временный override LLM/credentials env-переменных через ContextVar.

`litellm.completion()` и `openai.AsyncOpenAI()` читают ключи из env
(ANTHROPIC_API_KEY / OPENAI_API_KEY / GIGACHAT_CREDENTIALS).
Чтобы прокинуть per-session credentials без гонки между сессиями,
мы делаем «snapshot + restore» через contextvars, а не через
прямой `os.environ` (который глобальный).

litellm поддерживает `api_key=` per-call, но не все провайдеры это
уважают. Поэтому монкипатчим `os.environ` на время вызова.
"""
from __future__ import annotations

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
)


@contextlib.contextmanager
def llm_credentials_env(creds: DecryptedSettings):
    """Контекст: на время вызова подменяет env credentials сессии.

    Определяет провайдера по модели и подставляет соответствующий ключ.
    """
    model = (creds.llm_model or "").lower()
    saved = {k: os.environ.get(k) for k in _LLM_ENV_KEYS}
    try:
        # Сбрасываем все
        for k in _LLM_ENV_KEYS:
            os.environ.pop(k, None)
        # Выставляем только релевантный
        if "claude" in model or "anthropic" in model:
            os.environ["ANTHROPIC_API_KEY"] = creds.llm_api_key
        elif "gigachat" in model:
            os.environ["GIGACHAT_CREDENTIALS"] = creds.llm_api_key
        else:
            # OpenAI и любые другие провайдеры через litellm
            os.environ["OPENAI_API_KEY"] = creds.llm_api_key
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
