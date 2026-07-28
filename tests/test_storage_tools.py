"""Тесты storage-инструментов с моком БД.

Покрывают aqr/tools/storage.py и aqr/registry/store.py методы,
которые обычно требуют живую БД.
"""
from __future__ import annotations

import uuid
from datetime import UTC
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture(autouse=True)
def _stable_secret(monkeypatch):
    monkeypatch.setenv("AQR_SESSION_SECRET", "test-secret-padded-to-32-bytes-base64==")


@pytest.fixture
def with_credentials():
    """Устанавливает credentials в ContextVar, очищает на teardown."""
    from aqr.agent.context import reset_credentials, set_credentials
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

# ── Fake models ──────────────────────────────────────────────────


class _FakeHypothesis:
    def __init__(self, **kwargs):
        self.id = kwargs.get("id", uuid.uuid4())
        self.run_id = kwargs.get("run_id", uuid.uuid4())
        self.family = kwargs.get("family", "momentum")
        self.ticker = kwargs.get("ticker", "SBER")
        self.config_json = kwargs.get("config_json", {})
        self.dsr = kwargs.get("dsr")
        self.pbo = kwargs.get("pbo")
        self.cpcv = kwargs.get("cpcv")
        self.sharpe = kwargs.get("sharpe")
        self.max_drawdown = kwargs.get("max_drawdown")
        self.is_valid = kwargs.get("is_valid", False)
        self.embedding = kwargs.get("embedding")


class _FakeRun:
    def __init__(self, **kwargs):
        from datetime import datetime
        self.id = kwargs.get("id", uuid.uuid4())
        self.goal = kwargs.get("goal", "test goal")
        self.session_id = kwargs.get("session_id", "default")
        self.status = kwargs.get("status", "done")
        self.summary_metrics = kwargs.get("summary_metrics", {})
        self.created_at = kwargs.get(
            "created_at", datetime(2024, 1, 1, tzinfo=UTC),
        )


# ── Фикстуры для мока БД ────────────────────────────────────────


@pytest.fixture
def mock_db_and_store(monkeypatch):
    """Мок _async_session_factory и RegistryStore на уровне обоих модулей."""

    class _FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def commit(self):
            pass

    factory = type("_F", (), {"__call__": lambda self: _FakeSession()})()

    # Патчим db._async_session_factory
    from aqr import db as db_mod
    monkeypatch.setattr(db_mod, "_async_session_factory", factory)

    # Патчим storage._async_session_factory (импортирован в namespace)
    from aqr.tools import storage as storage_mod
    monkeypatch.setattr(storage_mod, "_async_session_factory", factory)

    # Создаём мок RegistryStore
    mock_store = MagicMock()

    from aqr.registry import store as store_mod
    monkeypatch.setattr(store_mod, "RegistryStore", lambda db: mock_store)

    # storage.py импортирует RegistryStore в свой namespace — патчим там тоже
    from aqr.tools import storage as storage_mod_inner
    monkeypatch.setattr(storage_mod_inner, "RegistryStore", lambda db: mock_store)

    return mock_store


# ── get_run ──────────────────────────────────────────────────────


class TestGetRun:
    @pytest.mark.asyncio
    async def test_get_run_returns_dict_when_found(self, mock_db_and_store):
        from aqr.tools import storage

        fake_run = _FakeRun(
            id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
            goal="проверь momentum",
            summary_metrics={"n_tested": 20, "n_survived_dsr": 5, "portfolio_pbo": 0.3},
        )
        fake_hyp = _FakeHypothesis(
            family="momentum", ticker="SBER",
            config_json={"fast": 10, "slow": 50},
            dsr=0.95, cpcv=0.7, sharpe=1.2, max_drawdown=-0.1, is_valid=True,
        )

        mock_db_and_store.get_run = AsyncMock(return_value=fake_run)
        mock_db_and_store.list_hypotheses_by_run = AsyncMock(return_value=[fake_hyp])

        result = await storage.get_run(str(fake_run.id))

        assert result is not None
        assert result["goal"] == "проверь momentum"
        assert result["status"] == "done"
        assert result["summary_metrics"]["n_tested"] == 20
        assert len(result["hypotheses"]) == 1
        assert result["hypotheses"][0]["family"] == "momentum"

    @pytest.mark.asyncio
    async def test_get_run_returns_none_when_not_found(self, mock_db_and_store):
        from aqr.tools import storage

        mock_db_and_store.get_run = AsyncMock(return_value=None)

        result = await storage.get_run(str(uuid.uuid4()))
        assert result is None


# ── compare_runs ─────────────────────────────────────────────────


class TestCompareRuns:
    @pytest.mark.asyncio
    async def test_compare_returns_delta(self, mock_db_and_store):
        from aqr.tools import storage

        run_a = _FakeRun(summary_metrics={"n_survived_dsr": 2, "portfolio_pbo": 0.5})
        run_b = _FakeRun(summary_metrics={"n_survived_dsr": 7, "portfolio_pbo": 0.3})

        mock_db_and_store.get_run = AsyncMock(side_effect=[run_a, run_b])

        result = await storage.compare_runs(str(uuid.uuid4()), str(uuid.uuid4()))

        assert "run_a" in result
        assert "run_b" in result
        assert result["delta"]["n_survived_dsr"] == 5
        assert abs(result["delta"]["portfolio_pbo"] - (-0.2)) < 0.001

    @pytest.mark.asyncio
    async def test_compare_returns_error_when_missing(self, mock_db_and_store):
        from aqr.tools import storage

        mock_db_and_store.get_run = AsyncMock(return_value=None)

        result = await storage.compare_runs(str(uuid.uuid4()), str(uuid.uuid4()))
        assert "error" in result


# ── list_runs ────────────────────────────────────────────────────


class TestListRuns:
    @pytest.mark.asyncio
    async def test_list_runs_returns_list(self, mock_db_and_store):
        from aqr.tools import storage

        runs = [
            _FakeRun(goal="g1", status="done", summary_metrics={"n_tested": 10}),
            _FakeRun(goal="g2", status="error", summary_metrics=None),
        ]

        mock_db_and_store.list_runs_by_session = AsyncMock(return_value=runs)

        result = await storage.list_runs(session_id="test", limit=5)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["goal"] == "g1"
        assert result[1]["status"] == "error"


# ── search_similar_hypotheses / find_duplicates ──────────────────


class TestSearchSimilarHypotheses:
    @pytest.mark.asyncio
    async def test_returns_similar_above_threshold(
        self, mock_db_and_store, monkeypatch, with_credentials
    ):
        from aqr.registry.embeddings import Embedder
        from aqr.tools import storage

        fake_hyp = _FakeHypothesis(
            family="momentum", ticker="SBER",
            dsr=0.95, sharpe=1.2, is_valid=True,
        )

        mock_db_and_store.search_by_text = AsyncMock(return_value=[
            (fake_hyp, 0.96),
            (fake_hyp, 0.55),  # ниже threshold 0.7 — отсеется
        ])

        async def fake_embed(self, text):
            return [0.0] * 1536

        monkeypatch.setattr(Embedder, "embed", fake_embed)

        result = await storage.search_similar_hypotheses(
            text="momentum on SBER", threshold=0.7, limit=5,
        )
        assert len(result) == 1
        assert result[0]["family"] == "momentum"
        assert result[0]["similarity"] == 0.96

    @pytest.mark.asyncio
    async def test_search_returns_empty_when_db_fails(self, monkeypatch, with_credentials):
        from aqr import db as db_mod
        from aqr.tools import storage
        from aqr.tools import storage as storage_mod

        class _BrokenFactory:
            def __call__(self):
                raise RuntimeError("DB down")

        monkeypatch.setattr(db_mod, "_async_session_factory", _BrokenFactory())
        monkeypatch.setattr(storage_mod, "_async_session_factory", _BrokenFactory())

        import pytest
        # Strict mode (Phase 6): search_similar_hypotheses now raises on DB failure
        # instead of returning empty list. Bug-for-bug backwards compatibility dropped.
        with pytest.raises(RuntimeError, match="DB down"):
            await storage.search_similar_hypotheses(text="x", threshold=0.5)

    @pytest.mark.asyncio
    async def test_find_duplicates_works(
        self, mock_db_and_store, monkeypatch, with_credentials
    ):
        """find_duplicates — обёртка над search_similar с высоким порогом."""
        from aqr.registry.embeddings import Embedder
        from aqr.tools import storage

        fake_hyp = _FakeHypothesis(family="momentum", ticker="SBER", dsr=0.95)

        mock_db_and_store.search_by_text = AsyncMock(return_value=[(fake_hyp, 0.95)])

        async def fake_embed(self, text):
            return [0.0] * 1536

        monkeypatch.setattr(Embedder, "embed", fake_embed)

        result = await storage.find_duplicates(text="momentum SBER", threshold=0.92)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["family"] == "momentum"
