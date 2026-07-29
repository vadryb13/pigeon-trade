"""Генерация эмбеддингов для гипотез.

OpenAI-совместимый API (text-embedding-3-small, 1536d).
Строгий режим: API-ключ обязателен (из per-session credentials через
ContextVar, либо явный параметр). Без ключа → raise. Без fallback.

Поддерживает OPENAI_BASE_URL для кастомных провайдеров
(совместимых с OpenAI-клиентом).
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import AsyncOpenAI


from ..types import EMBEDDING_DIM  # noqa: F401 — реэкспорт для обратной совместимости


class Embedder:
    """Обёртка над OpenAI-совместимым embeddings-API. Lazy init клиента."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._base_url = base_url
        self._client: AsyncOpenAI | None = None

        # Если ключ не передан явно — берём из per-session ContextVar
        if self._api_key is None:
            self._api_key = self._api_key_from_context()
        # Если всё ещё нет — из env как fallback
        if self._api_key is None:
            self._api_key = os.environ.get("OPENAI_API_KEY")

        # Если base_url не передан — читаем из env
        if self._base_url is None:
            self._base_url = os.environ.get("OPENAI_BASE_URL")

    @staticmethod
    def _api_key_from_context() -> str | None:
        """Получить OpenAI API key из ContextVar credentials."""
        try:
            from aqr.graph.context import current_credentials

            creds = current_credentials()
            if creds and creds.openai_api_key:
                return creds.openai_api_key
        except (ImportError, RuntimeError):
            pass
        return None

    @staticmethod
    def hypothesis_to_text(family: str, ticker: str, params: dict) -> str:
        """Каноническое текстовое представление гипотезы для эмбеддинга."""
        params_str = ", ".join(
            f"{k}={v}" for k, v in sorted(params.items())
        )
        return f"{family} on {ticker}: {params_str}".strip(": ")

    async def embed(self, text: str) -> list[float]:
        """Эмбеддинг строки через OpenAI. Raise на любой ошибке."""
        return await self._embed_openai(text)

    async def embed_hypothesis(
        self, family: str, ticker: str, params: dict
    ) -> list[float]:
        text = self.hypothesis_to_text(family, ticker, params)
        return await self.embed(text)

    async def _embed_openai(self, text: str) -> list[float]:
        """Через OpenAI-совместимый API (lazy import)."""
        if self._client is None:
            from openai import AsyncOpenAI

            if not self._api_key:
                raise RuntimeError(
                    "Embedder: no API key available. Set OPENAI_API_KEY env var "
                    "or configure via /chat/{token}/settings."
                )

            client_kwargs = {"api_key": self._api_key}
            if self._base_url:
                client_kwargs["base_url"] = self._base_url
            self._client = AsyncOpenAI(**client_kwargs)

        resp = await self._client.embeddings.create(
            model=self.model,
            input=text,
        )
        return list(resp.data[0].embedding)

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Косинусное сходство двух векторов (float)."""
        import numpy as np

        va = np.asarray(a, dtype=np.float32)
        vb = np.asarray(b, dtype=np.float32)
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        if denom == 0:
            return 0.0
        return float(np.dot(va, vb) / denom)
