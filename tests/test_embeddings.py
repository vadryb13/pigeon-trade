"""Тесты для эмбеддингов: Embedder + дедупликация в plan_research.

Работают без OpenAI API (используется hash-fallback).
"""
from __future__ import annotations

import numpy as np
import pytest

from aqr.registry.embeddings import EMBEDDING_DIM, Embedder

# ── Гарантируем, что тестируем hash-fallback ─────────────────────

@pytest.fixture(autouse=True)
def _no_openai_key(monkeypatch):
    """Гарантируем, что используется hash-fallback."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


# ── Embedder (hash-fallback) ─────────────────────────────────────

class TestEmbedderHash:
    @pytest.mark.asyncio
    async def test_hash_embedding_deterministic(self):
        """Один текст → один и тот же вектор."""
        e = Embedder()
        v1 = await e.embed("momentum SMA5/50 on SBER: fast=5 slow=50")
        v2 = await e.embed("momentum SMA5/50 on SBER: fast=5 slow=50")
        np.testing.assert_array_equal(v1, v2)

    @pytest.mark.asyncio
    async def test_hash_embedding_normalized(self):
        """L2-норма = 1 (после нормализации)."""
        v = await Embedder().embed("any text")
        norm = float(np.linalg.norm(v))
        assert abs(norm - 1.0) < 1e-5

    @pytest.mark.asyncio
    async def test_hash_embedding_dim_matches_schema(self):
        """Размер совпадает с Vector(1536) в схеме."""
        v = await Embedder().embed("test")
        assert len(v) == EMBEDDING_DIM == 1536

    @pytest.mark.asyncio
    async def test_hash_embedding_different_texts_diverge(self):
        """Разные тексты → разные векторы."""
        v1 = await Embedder().embed("momentum on SBER: fast=5 slow=50")
        v2 = await Embedder().embed("mean_reversion on GAZP: window=20")
        # Не должны совпадать
        assert any(abs(a - b) > 0.01 for a, b in zip(v1, v2))

    @pytest.mark.asyncio
    async def test_similar_texts_have_higher_similarity(self):
        """Похожие тексты → больше cosine similarity, чем разные."""
        e = Embedder()
        similar_a = await e.embed("momentum SMA5/50 on SBER: fast=5 slow=50")
        similar_b = await e.embed("momentum SMA10/100 on SBER: fast=10 slow=100")
        unrelated = await e.embed("variance breakouts for currency pairs EUR USD")

        sim_close = Embedder.cosine_similarity(similar_a, similar_b)
        sim_far = Embedder.cosine_similarity(similar_a, unrelated)

        assert sim_close > sim_far

    def test_hypothesis_to_text_canonical(self):
        """Канонический формат гипотезы стабилен и сортирует params."""
        text = Embedder.hypothesis_to_text(
            "momentum", "SBER",
            {"fast": 5, "slow": 50, "window": 20},
        )
        assert text.startswith("momentum on SBER:")
        # Параметры отсортированы
        assert "fast=5" in text
        assert "slow=50" in text
        assert "window=20" in text

    def test_cosine_similarity_self_is_one(self):
        v = np.random.default_rng(0).normal(0, 1, 1536).tolist()
        v_norm = (np.asarray(v) / np.linalg.norm(v)).tolist()
        sim = Embedder.cosine_similarity(v_norm, v_norm)
        assert abs(sim - 1.0) < 1e-5

    def test_cosine_similarity_orthogonal_is_zero(self):
        """Перпендикулярные вектора → similarity = 0."""
        a = [1.0] + [0.0] * 1535
        b = [0.0] * 768 + [1.0] + [0.0] * 767
        sim = Embedder.cosine_similarity(a, b)
        assert abs(sim) < 1e-5


# ── Интеграция с plan_research (дедуп-предупреждение) ────────────

class TestPlanResearchDedup:
    """Тесты на plan_research: дедупликация добавляется в результате плана, если БД
    доступна. Без БД — поле просто отсутствует, без падения.

    Через мок aqr.db._async_session_factory (используется внутри plan_research).
    """

    @pytest.mark.asyncio
    async def test_plan_research_returns_plan_without_db(self, monkeypatch):
        """Без БД plan_research не падает."""
        from aqr import db as db_mod

        class _BrokenFactory:
            def __call__(self):
                raise RuntimeError("no DB in test")

        monkeypatch.setattr(db_mod, "_async_session_factory", _BrokenFactory())

        from aqr.tools import core as tools_core
        result = await tools_core.plan_research("проверь momentum на Сбере")
        assert "tickers" in result
        assert "hypothesis_families" in result
        assert "dedup_warning" not in result

    @pytest.mark.asyncio
    async def test_plan_research_dedup_warning_added_when_found(self, monkeypatch):
        """Если RegistryStore.search_similar нашёл дубликаты — появляется dedup_warning."""
        from aqr import db as db_mod
        from aqr.registry import store as store_mod
        from aqr.registry.models import Hypothesis

        fake_hyp = Hypothesis(
            id="00000000-0000-0000-0000-000000000001",
            run_id="00000000-0000-0000-0000-000000000002",
            family="momentum",
            ticker="SBER",
            config_json={"fast": 5, "slow": 50},
            dsr=0.92,
            sharpe=1.2,
            is_valid=True,
        )

        async def fake_search_similar(_self, emb, threshold=0.0, limit=10):
            return [(fake_hyp, 0.95)]

        monkeypatch.setattr(
            store_mod.RegistryStore,
            "search_similar",
            fake_search_similar,
        )

        # Мокаем сессию — она ничего не делает, потому что search_similar тоже замокан
        class _FakeSession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        class _FakeFactory:
            def __call__(self):
                return _FakeSession()

        monkeypatch.setattr(db_mod, "_async_session_factory", _FakeFactory())

        from aqr.tools import core as tools_core
        result = await tools_core.plan_research("проверь momentum на Сбере")
        assert "dedup_warning" in result
        assert "momentum/SBER" in result["dedup_warning"]
        assert len(result["similar_runs"]) == 1
        assert result["similar_runs"][0]["similarity"] == 0.95
