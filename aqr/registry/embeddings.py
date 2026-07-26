"""Генерация эмбеддингов для гипотез.

Основной путь: OpenAI `text-embedding-3-small` (1536d, $0.02/1M tokens).
Fallback: детерминистический hash-вектор (при отсутствии OPENAI_API_KEY
или ошибке API). Размерность совпадает с Vector(1536) в `Hypothesis.embedding`.
"""

from __future__ import annotations

import hashlib
import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from openai import AsyncOpenAI


EMBEDDING_DIM = 1536


class Embedder:
    """Обёртка над embeddings-API. Lazy init клиента."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client: AsyncOpenAI | None = None

    def has_api(self) -> bool:
        """True, если есть API-ключ И пакет `openai` установлен.

        Проверяем `import openai` чтобы не вернуть True, когда
        ключ есть, но пакет не установлен (тогда первый вызов
        упадёт с ImportError, который мы глотаем в `embed()` —
        выглядит как молчаливый fallback на hash).
        """
        if not self._api_key:
            return False
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return True

    @staticmethod
    def hypothesis_to_text(family: str, ticker: str, params: dict) -> str:
        """Каноническое текстовое представление гипотезы для эмбеддинга."""
        params_str = ", ".join(
            f"{k}={v}" for k, v in sorted(params.items())
        )
        return f"{family} on {ticker}: {params_str}".strip(": ")

    async def embed(self, text: str) -> list[float]:
        """Эмбеддинг строки. OpenAI если есть ключ, иначе hash-fallback."""
        if self.has_api():
            try:
                return await self._embed_openai(text)
            except Exception:
                pass
        return self.hash_embedding(text)

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
    def hash_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
        """Детерминистический SHA256-based вектор длиной dim.

        Расширяем хэш через повторение до dim байт, нормализуем до unit-вектора.
        Семантики нет, но гарантируется:
        - одинаковый текст → одинаковый вектор (для дедупликации)
        - разный текст → разные векторы (с высокой вероятностью)
        - L2 норма = 1.0
        """
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # Каждый sha256-дайджест = 32 байта; нам нужно 1536/32 = 48 повторений
        repeats = (dim // 32) + 1
        raw = (digest * repeats)[:dim]
        arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 127.5) / 127.5
        norm = float(np.linalg.norm(arr))
        if norm > 0:
            arr = arr / norm
        return arr.tolist()

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Косинусное сходство двух векторов (float)."""
        va = np.asarray(a, dtype=np.float32)
        vb = np.asarray(b, dtype=np.float32)
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        if denom == 0:
            return 0.0
        return float(np.dot(va, vb) / denom)
