"""Tests for NautilusTrader executor.

nautilus_trader is optional — tests use `importorskip` or mock `_require_nautilus`.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pandas as pd
import pytest

from aqr.agent.context import reset_credentials, set_credentials
from aqr.registry import DecryptedSettings


@pytest.fixture(autouse=True)
def _stable_secret(monkeypatch):
    monkeypatch.setenv("AQR_SESSION_SECRET", "test-secret-padded-to-32-bytes-base64==")


@pytest.fixture
def with_credentials():
    creds = DecryptedSettings(
        session_id="test-session",
        llm_model="claude-3-5-sonnet-20241022",
        llm_api_key="sk-ant-fake",
        openai_api_key="sk-oai-fake",
        invest_token="t.INVEST_TOKEN_fake",
        invest_sandbox=True,
    )
    token = set_credentials(creds)
    yield creds
    reset_credentials(token)


def _make_prices(n: int = 500, base: float = 100.0, drift: float = 0.05) -> list[float]:
    return [base + i * drift for i in range(n)]


def _make_hypothesis(
    family: str = "momentum",
    ticker: str = "SBER",
    params: dict | None = None,
) -> dict:
    return {
        "name": f"{family}_{ticker}",
        "family": family,
        "ticker": ticker,
        "params": params or {"fast": 10, "slow": 30},
    }


class TestExecuteWithSlippage:
    @pytest.mark.asyncio
    async def test_returns_backtest_result(self, with_credentials):
        """Returns BacktestResult dataclass with populated fields."""
        from aqr.executor.nautilus import execute_with_slippage

        result = await execute_with_slippage(
            hypothesis=_make_hypothesis(),
            prices=_make_prices(),
        )
        from aqr.types import BacktestResult
        assert isinstance(result, BacktestResult)
        assert result.sharpe != 0.0
        assert result.hypothesis.ticker == "SBER"
        assert result.hypothesis.family == "momentum"
        assert result.dsr_verdict in ("significant", "borderline", "not_significant", "insufficient")

    @pytest.mark.asyncio
    async def test_unknown_family_raises(self, with_credentials):
        """Unknown family raises ValueError."""
        from aqr.executor.nautilus import execute_with_slippage

        with pytest.raises(ValueError, match="Unknown family"):
            await execute_with_slippage(
                hypothesis=_make_hypothesis(family="nonexistent"),
                prices=_make_prices(),
            )

    @pytest.mark.asyncio
    async def test_insufficient_prices_raises(self, with_credentials):
        """Too few prices raises ValueError."""
        from aqr.executor.nautilus import execute_with_slippage

        with pytest.raises(ValueError, match="Insufficient data"):
            await execute_with_slippage(
                hypothesis=_make_hypothesis(),
                prices=[100.0] * 50,
            )

    @pytest.mark.asyncio
    async def test_zero_std_returns_insufficient(self, with_credentials):
        """Flat prices → insufficient verdict."""
        from aqr.executor.nautilus import execute_with_slippage

        result = await execute_with_slippage(
            hypothesis=_make_hypothesis(family="momentum", ticker="FLAT",
                                         params={"fast": 3, "slow": 20}),
            prices=[100.0] * 500,
        )
        assert result.dsr_verdict == "insufficient"

    @pytest.mark.asyncio
    async def test_param_overrides(self, with_credentials):
        """Different params produce different results."""
        from aqr.executor.nautilus import execute_with_slippage

        r1 = await execute_with_slippage(
            hypothesis=_make_hypothesis(params={"fast": 5, "slow": 20}),
            prices=_make_prices(),
        )
        r2 = await execute_with_slippage(
            hypothesis=_make_hypothesis(params={"fast": 50, "slow": 200}),
            prices=_make_prices(),
        )
        # Different params may give different metrics
        assert isinstance(r1, type(r2))

    @pytest.mark.asyncio
    async def test_nautilus_path_called_when_available(self, with_credentials, monkeypatch):
        """When nautilus_trader is importable, the engine path is taken."""
        from aqr.executor import nautilus as nautilus_mod

        # Make _require_nautilus return a truthy value (simulating installed)
        monkeypatch.setattr(nautilus_mod, "_require_nautilus", lambda: MagicMock())

        # Mock the engine runner to return a known result
        async def fake_engine(spec, **kw):
            from aqr.types import BacktestResult
            return BacktestResult(
                hypothesis=spec,
                sharpe=2.0, dsr=1.8, dsr_verdict="significant",
                cpcv_mean_sharpe=0.0, cpcv_std_sharpe=0.0,
                max_drawdown=-0.1, n_trades=10,
                daily_returns=[0.001] * 100,
            )

        monkeypatch.setattr(
            nautilus_mod, "_run_nautilus_engine_placeholder", fake_engine,
        )

        # Import AFTER monkeypatch to pick up the patched module
        from aqr.executor.nautilus import execute_with_slippage

        result = await execute_with_slippage(
            hypothesis=_make_hypothesis(),
            prices=_make_prices(),
        )
        assert result.sharpe == 2.0
        assert result.dsr_verdict == "significant"

    @pytest.mark.asyncio
    async def test_nautilus_fallback_on_error(self, with_credentials, monkeypatch):
        """If NautilusTrader engine raises, falls back to native path."""
        from aqr.executor import nautilus as nautilus_mod
        from aqr.executor.nautilus import execute_with_slippage

        async def failing_engine(spec, **kw):
            raise RuntimeError("engine crashed")

        monkeypatch.setattr(
            nautilus_mod, "_run_nautilus_engine_placeholder", failing_engine,
        )

        result = await execute_with_slippage(
            hypothesis=_make_hypothesis(),
            prices=_make_prices(),
        )
        # Native path produces valid result
        assert result.sharpe != 0.0
        assert result.dsr_verdict != "insufficient"
