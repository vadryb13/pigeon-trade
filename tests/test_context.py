"""Тесты для aqr/agent/context.py — SessionContext.

Покрывают методы get_recent_runs, get_best_strategy, get_untested_combos
с моком БД.
"""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

from conftest import BrokenFactory, FakeSession

import pytest


class _FakeHyp:
    def __init__(self, **kw):
        self.id = kw.get("id", uuid.uuid4())
        self.run_id = kw.get("run_id", uuid.uuid4())
        self.family = kw.get("family", "momentum")
        self.ticker = kw.get("ticker", "SBER")
        self.config_json = kw.get("config_json", {"fast": 5, "slow": 50})
        self.dsr = kw.get("dsr")
        self.sharpe = kw.get("sharpe")
        self.is_valid = kw.get("is_valid", False)


class _FakeRun:
    def __init__(self, **kw):
        self.id = kw.get("id", uuid.uuid4())
        self.goal = kw.get("goal", "test")
        self.status = kw.get("status", "done")
        self.summary_metrics = kw.get("summary_metrics", {})


@pytest.fixture
def mock_db(monkeypatch):
    """Мок БД на уровне agent/context.py и db. RegistryStore тоже мокается."""
    factory = type("_F", (), {"__call__": lambda self: FakeSession()})()

    from aqr import session as db_mod
    monkeypatch.setattr(db_mod, "async_session_factory", factory)
    from aqr.graph import context as ctx_mod
    monkeypatch.setattr(ctx_mod, "async_session_factory", factory)

    # Patch RegistryStore (импортирован в context.py через `from ..registry import RegistryStore`)
    mock_store = MagicMock()
    from aqr.registry import store as store_mod
    monkeypatch.setattr(store_mod, "RegistryStore", lambda db: mock_store)
    monkeypatch.setattr(ctx_mod, "RegistryStore", lambda db: mock_store)

    return mock_store


# ── get_recent_runs ──────────────────────────────────────────────


class TestGetRecentRuns:
    @pytest.mark.asyncio
    async def test_returns_formatted_dicts(self, mock_db):
        from aqr.graph.context import SessionContext

        runs = [
            _FakeRun(goal="g1", summary_metrics={"n_tested": 10, "n_survived_dsr": 2, "portfolio_pbo": 0.3}),
            _FakeRun(goal="g2", summary_metrics={}),
        ]
        mock_db.list_runs_by_session = AsyncMock(return_value=runs)

        ctx = SessionContext("sess-1")
        result = await ctx.get_recent_runs(limit=5)

        assert len(result) == 2
        assert result[0]["goal"] == "g1"
        assert result[0]["status"] == "done"
        assert result[0]["metrics"]["n_tested"] == 10
        assert "id" in result[0]


# ── get_best_strategy ────────────────────────────────────────────


class TestGetBestStrategy:
    @pytest.mark.asyncio
    async def test_finds_max_dsr_hypothesis(self, mock_db):
        from aqr.graph.context import SessionContext

        run_a = _FakeRun(goal="r1")
        run_b = _FakeRun(goal="r2")
        hyps_a = [_FakeHyp(dsr=0.7, sharpe=1.0, family="momentum")]
        hyps_b = [_FakeHyp(dsr=0.95, sharpe=1.5, family="mean_reversion", ticker="GAZP")]

        mock_db.list_runs_by_session = AsyncMock(return_value=[run_a, run_b])
        # B11: батч-метод возвращает dict[run_id, list[Hypothesis]].
        mock_db.list_hypotheses_by_runs = AsyncMock(
            return_value={run_a.id: hyps_a, run_b.id: hyps_b}
        )

        ctx = SessionContext("sess-1")
        best = await ctx.get_best_strategy()

        assert best is not None
        assert best["dsr"] == 0.95
        assert best["family"] == "mean_reversion"
        assert best["ticker"] == "GAZP"

    @pytest.mark.asyncio
    async def test_returns_none_when_no_runs(self, mock_db):
        from aqr.graph.context import SessionContext

        mock_db.list_runs_by_session = AsyncMock(return_value=[])

        ctx = SessionContext("sess-empty")
        best = await ctx.get_best_strategy()
        assert best is None


# ── get_untested_combos ──────────────────────────────────────────


class TestGetUntestedCombos:
    @pytest.mark.asyncio
    async def test_suggests_uncovered_family_on_known_ticker(self, mock_db):
        from aqr.graph.context import SessionContext

        run = _FakeRun()
        # Все комбинации кроме (mean_reversion, SBER) уже проверены
        hyps = [
            _FakeHyp(family="momentum", ticker="SBER"),
            _FakeHyp(family="breakout", ticker="SBER"),
            _FakeHyp(family="volatility", ticker="SBER"),
        ]
        mock_db.list_runs_by_session = AsyncMock(return_value=[run])
        # B11: батч-метод.
        mock_db.list_hypotheses_by_runs = AsyncMock(return_value={run.id: hyps})

        ctx = SessionContext("sess-1")
        suggestions = await ctx.get_untested_combos()

        assert any(
            s["family"] == "mean_reversion" and s["ticker"] == "SBER"
            for s in suggestions
        )

    @pytest.mark.asyncio
    async def test_returns_empty_when_db_fails(self, monkeypatch):
        from aqr import session as db_mod
        from aqr.graph.context import SessionContext

        monkeypatch.setattr(db_mod, "async_session_factory", BrokenFactory())

        ctx = SessionContext("sess")
        suggestions = await ctx.get_untested_combos()
        assert suggestions == []


# ── build_context_prompt с реальным контентом ───────────────────


class TestBuildContextPrompt:
    @pytest.mark.asyncio
    async def test_includes_recent_runs_and_best_strategy(self, mock_db):
        from aqr.graph.context import SessionContext

        run = _FakeRun(goal="проверь momentum", summary_metrics={
            "n_tested": 20, "n_survived_dsr": 5, "portfolio_pbo": 0.3,
        })
        hyp = _FakeHyp(family="momentum", ticker="SBER", dsr=0.92, sharpe=1.2)
        mock_db.list_runs_by_session = AsyncMock(return_value=[run])
        mock_db.list_hypotheses_by_runs = AsyncMock(return_value={run.id: [hyp]})

        ctx = SessionContext("sess-1")
        prompt = await ctx.build_context_prompt()

        assert "проверь momentum" in prompt
        assert "Лучшая стратегия" in prompt
        assert "momentum" in prompt
        assert "SBER" in prompt

    @pytest.mark.asyncio
    async def test_includes_untested_section(self, mock_db):
        """Когда есть непроверенные комбинации — они попадают в prompt."""
        from aqr.graph.context import SessionContext

        # 1 run, где проверены только momentum/SBER — остальные семейства белые пятна
        run = _FakeRun()
        hyps = [_FakeHyp(family="momentum", ticker="SBER")]
        mock_db.list_runs_by_session = AsyncMock(return_value=[run])
        mock_db.list_hypotheses_by_runs = AsyncMock(return_value={run.id: hyps})

        ctx = SessionContext("sess")
        prompt = await ctx.build_context_prompt()

        assert "Непроверенные комбинации" in prompt
        assert "SBER" in prompt
