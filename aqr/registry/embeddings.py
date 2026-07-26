"""Генерация эмбеддингов для гипотез.

Только OpenAI `text-embedding-3-small` (1536d, $0.02/1M tokens).
Строгий режим: API-ключ обязателен (из per-session credentials через
ContextVar, либо явный параметр). Без ключа → raise. Без fallback.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import AsyncOpenAI


EMBEDDING_DIM = 1536


class Embedder:
    """Обёртка над OpenAI embeddings-API. Lazy init клиента."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._client: AsyncOpenAI | None = None

        # Если ключ не передан явно — берём из per-session ContextVar
        if self._api_key is None:
            self._api_key = self._api_key_from_context()

    @staticmethod
    def _api_key_from_context() -> str:
        """Получить OpenAI API key из ContextVar credentials."""
        from aqr.agent.context import current_credentials

        creds = current_credentials()
        if creds is None:
            raise RuntimeError(
                "Embedder: OPENAI_API_KEY not provided and no session "
                "credentials in context. Configure via /chat/{token}/settings."
            )
        return creds.openai_api_key

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
        """Через OpenAI API (lazy import)."""
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self._api_key)

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
