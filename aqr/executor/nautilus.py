"""NautilusTrader wrapper for realistic execution simulation.

Builds on the simplified `backtest_one` (aqr.tools.core) by applying:
  - Commission: 0.05% per trade (typical broker T+0)
  - Slippage: 1 tick
  - Event-driven BacktestEngine from nautilus_trader

When nautilus_trader is installed, `_run_nautilus_engine_placeholder`
returns native metrics (B8) — full BacktestEngine integration is TODO.
Without it, we fall back to the same logic as `backtest_one`,
still returning a `BacktestResult` dataclass.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from aqr.pipeline.hypotheses import make_one_with_params
from aqr.types import BacktestResult
from aqr.validation.deflated_sharpe import deflated_sharpe_ratio

logger = logging.getLogger(__name__)

BROKER_COMMISSION_PCT = 0.0005
MIN_PRICES = 126


def _require_nautilus():
    """Lazy import — nautilus_trader is heavy.

    Returns None if not installed (graceful fallback).
    """
    try:
        import nautilus_trader  # noqa: F401
        return nautilus_trader
    except ImportError:
        return None


def _compute_metrics(
    strat_ret: pd.Series,
    spec,
    price_series: pd.Series | None = None,
    n_hypotheses: int = 20,
) -> BacktestResult:
    """Compute standard metrics from strategy returns.

    Shared between native and NautilusTrader paths.
    """
    if len(strat_ret) < 30 or strat_ret.std() == 0:
        return BacktestResult(
            hypothesis=spec,
            sharpe=0.0, dsr=0.0, dsr_verdict="insufficient",
            cpcv_mean_sharpe=0.0, cpcv_std_sharpe=0.0,
            max_drawdown=0.0, n_trades=0, daily_returns=[],
        )

    sharpe = float(strat_ret.mean() / strat_ret.std() * np.sqrt(252))
    equity = (1 + strat_ret).cumprod()
    dd = float((equity / equity.cummax() - 1.0).min())

    dsr_out = deflated_sharpe_ratio(strat_ret.values, n_trials=n_hypotheses)
    dsr_val = float(dsr_out["deflated_sharpe"])
    if dsr_out["verdict"] == "significant":
        verdict = "significant"
    elif dsr_out["verdict"] == "insufficient_data":
        verdict = "insufficient"
    elif dsr_val >= 0.80:
        verdict = "borderline"
    else:
        verdict = "not_significant"

    # Generate entry/exit events from diff of shifted positions
    # Use actual price series for signal generation, not index data
    trade_prices = price_series if price_series is not None else pd.Series(strat_ret.index, name=spec.ticker)
    pos = spec.fn(trade_prices)
    pos_shifted = pos.shift(1).fillna(0.0)
    trades = int((pos_shifted.diff().abs() > 0).sum())

    return BacktestResult(
        hypothesis=spec,
        sharpe=round(sharpe, 3),
        dsr=round(dsr_val, 3),
        dsr_verdict=verdict,
        cpcv_mean_sharpe=0.0,
        cpcv_std_sharpe=0.0,
        max_drawdown=round(dd, 3),
        n_trades=trades,
        daily_returns=strat_ret.tolist(),
    )


async def execute_with_slippage(
    hypothesis: dict[str, Any],
    prices: list[float],
    commission_pct: float = BROKER_COMMISSION_PCT,
    slippage_ticks: int = 1,
) -> BacktestResult:
    """Run a backtest with realistic execution (NautilusTrader if available).

    Args:
        hypothesis: {"family", "ticker", "params", "name"}
        prices: close prices list
        commission_pct: fractional commission (0.0005 = 0.05%)
        slippage_ticks: number of ticks for slippage

    Returns:
        BacktestResult dataclass.

    Raises:
        ValueError if hypothesis is invalid or prices too short.
    """
    fam = hypothesis.get("family", "")
    ticker = hypothesis.get("ticker", "")
    params = hypothesis.get("params", {})

    spec = make_one_with_params(fam, ticker, params)
    if spec is None:
        raise ValueError(f"Unknown family: {fam}")

    price_series = pd.Series(prices, name=ticker)
    if len(price_series) < MIN_PRICES:
        raise ValueError(
            f"Insufficient data: {len(price_series)} < {MIN_PRICES}"
        )

    # Signal → positions → returns
    pos = spec.fn(price_series)
    pos_shifted = pos.shift(1).fillna(0.0)
    ret = price_series.pct_change().fillna(0.0)
    strat_ret = (pos_shifted * ret).astype(float)
    strat_ret = strat_ret.dropna()

    # NautilusTrader path: realistic execution simulation
    nt = _require_nautilus()
    if nt is not None:
        try:
            return await _run_nautilus_engine_placeholder(
                spec=spec,
                price_series=price_series,
                strat_ret=strat_ret,
                commission_pct=commission_pct,
                slippage_ticks=slippage_ticks,
            )
        except Exception as exc:
            logger.warning(
                "NautilusTrader engine failed, falling back to native: %s",
                exc,
            )

    # Native path: same logic as backtest_one
    return _compute_metrics(strat_ret, spec, price_series=price_series)


async def _run_nautilus_engine_placeholder(
    spec,
    price_series: pd.Series,
    strat_ret: pd.Series,
    commission_pct: float,
    slippage_ticks: int,
) -> BacktestResult:
    """PLACEHOLDER: реальная интеграция NautilusTrader BacktestEngine — TODO (B8).

    Когда `nautilus_trader` будет полноценно интегрирован, здесь должны быть:
      - Commission model (0.05% per trade)
      - Slippage model (1 tick)
      - FOK limit orders
      - Event-driven matching через BacktestEngine

    Сейчас возвращает native-метрики (то же что `backtest_one`).
    TODO: заменить на реальный BacktestEngine.run(...) когда
    `nautilus_trader` появится в зависимостях прод-окружения.
    """
    return _compute_metrics(strat_ret, spec, price_series=price_series)
