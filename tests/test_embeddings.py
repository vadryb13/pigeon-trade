"""Тесты Embedder — OpenAI embeddings с моком openai.AsyncOpenAI.

Без hash-fallback. Без OPENAI_API_KEY в ContextVar → raise.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _stable_secret(monkeypatch):
    monkeypatch.setenv("AQR_SESSION_SECRET", "test-secret-padded-to-32-bytes-base64==")


@pytest.fixture
def fake_openai(monkeypatch):
    """Подменяет openai.AsyncOpenAI().embeddings.create() фейком.

    Возвращает вектор размерности EMBEDDING_DIM = 1536 с уникальными
    значениями для разных текстов, чтобы cosine similarity работала.
    """
    from aqr.registry.embeddings import EMBEDDING_DIM

    _responses: dict[str, list[float]] = {}

    def make_embedding(text: str) -> list[float]:
        # Детерминированно из hash, но с разным seed для разных текстов
        import hashlib
        digest = hashlib.sha256(text.encode()).digest()
        repeats = (EMBEDDING_DIM // 32) + 1
        raw = (digest * repeats)[:EMBEDDING_DIM]
        return [(b / 255.0) for b in raw]

    class _FakeEmbeddingsAPI:
        def __init__(self, **kw):
            pass

        async def create(self, *, model, input):
            if isinstance(input, str):
                inputs = [input]
            else:
                inputs = input
            data = [
                MagicMock(embedding=make_embedding(t)) for t in inputs
            ]
            resp = MagicMock()
            resp.data = data
            return resp

    class _FakeAsyncOpenAI:
        def __init__(self, api_key=None, **kw):
            self._embeddings = _FakeEmbeddingsAPI()

        @property
        def embeddings(self):
            return self._embeddings

    fake_module = types.ModuleType("openai")
    fake_module.AsyncOpenAI = _FakeAsyncOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_module)

    return make_embedding


@pytest.fixture
def with_credentials():
    """Устанавливает credentials в ContextVar с OPENAI_API_KEY."""
    from aqr.graph.context import reset_credentials, set_credentials
    from aqr.registry import DecryptedSettings

    creds = DecryptedSettings(
        session_id="alice",
        llm_model="claude-3-5-sonnet-20241022",
        llm_api_key="sk-ant-fake",
        openai_api_key="sk-oai-fake",
        invest_token="t.INVEST_TOKEN_fake",
        invest_sandbox=True,
    )
    token = set_credentials(creds)
    yield creds
    reset_credentials(token)


class TestEmbedderRequiresKey:
    async def test_raises_without_api_key_and_without_credentials(self, monkeypatch):
        """Без api_key параметром и без ContextVar → RuntimeError при embed()."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from aqr.registry.embeddings import Embedder

        embedder = Embedder()
        with pytest.raises(RuntimeError, match="no API key available"):
            await embedder.embed("test")

    async def test_accepts_explicit_api_key(self, fake_openai):
        """Переданный api_key явно → ok, без credentials."""
        from aqr.registry.embeddings import Embedder

        embedder = Embedder(api_key="sk-explicit")
        v = await embedder.embed("test text")
        assert len(v) == 768

    async def test_uses_credentials_when_no_api_key(self, fake_openai, with_credentials):
        """Без параметра → берёт openai_api_key из ContextVar."""
        from aqr.registry.embeddings import Embedder

        embedder = Embedder()
        v = await embedder.embed("test text")
        assert len(v) == 768


class TestEmbedderEmbed:
    async def test_returns_correct_dim(self, fake_openai, with_credentials):
        from aqr.registry.embeddings import EMBEDDING_DIM, Embedder

        embedder = Embedder()
        v = await embedder.embed("any text")
        assert len(v) == EMBEDDING_DIM == 768

    async def test_deterministic_for_same_text(self, fake_openai, with_credentials):
        from aqr.registry.embeddings import Embedder

        embedder = Embedder()
        v1 = await embedder.embed("momentum SMA5/50 on SBER: fast=5 slow=50")
        v2 = await embedder.embed("momentum SMA5/50 on SBER: fast=5 slow=50")
        assert v1 == v2

    async def test_different_texts_different_vectors(
        self, fake_openai, with_credentials
    ):
        from aqr.registry.embeddings import Embedder

        embedder = Embedder()
        v1 = await embedder.embed("momentum on SBER: fast=5 slow=50")
        v2 = await embedder.embed("mean_reversion on GAZP: window=20")
        # Разные тексты → разные векторы (хотя бы один элемент разный)
        assert any(abs(a - b) > 0.01 for a, b in zip(v1, v2))


class TestEmbedderHypothesisText:
    def test_hypothesis_to_text_format(self):
        from aqr.registry.embeddings import Embedder

        text = Embedder.hypothesis_to_text(
            "momentum", "SBER", {"fast": 5, "slow": 50},
        )
        assert text == "momentum on SBER: fast=5, slow=50"

    def test_hypothesis_to_text_sorted_params(self):
        """Параметры сортируются для детерминированности."""
        from aqr.registry.embeddings import Embedder

        text1 = Embedder.hypothesis_to_text("momentum", "SBER", {"fast": 5, "slow": 50})
        text2 = Embedder.hypothesis_to_text("momentum", "SBER", {"slow": 50, "fast": 5})
        assert text1 == text2

    def test_hypothesis_to_text_empty_params(self):
        from aqr.registry.embeddings import Embedder

        text = Embedder.hypothesis_to_text("momentum", "SBER", {})
        assert text == "momentum on SBER"


class TestEmbedderCosineSimilarity:
    async def test_similar_texts_have_higher_similarity(
        self, fake_openai, with_credentials
    ):
        """Похожие тексты → больше cosine similarity, чем разные."""
        from aqr.registry.embeddings import Embedder

        embedder = Embedder()
        similar_a = await embedder.embed(
            "momentum SMA5/50 on SBER: fast=5 slow=50"
        )
        similar_b = await embedder.embed(
            "momentum SMA10/100 on SBER: fast=10 slow=100"
        )
        unrelated = await embedder.embed(
            "variance breakouts for currency pairs EUR USD"
        )

        sim_close = Embedder.cosine_similarity(similar_a, similar_b)
        sim_far = Embedder.cosine_similarity(similar_a, unrelated)

        assert sim_close > sim_far
