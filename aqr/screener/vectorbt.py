"""VectorBT-based fast hypothesis screener.

Iterates over a parameter grid (fast x slow SMA windows) and returns
the top-N combinations by Sharpe ratio. Used for idea screening
before committing to a single hypothesis (the existing
`aqr.tools.core.backtest_one` only runs ONE combo at a time).

Why VectorBT:
- Numba-accelerated; 100+ combos over 4y daily data in seconds
- Vectorized backtester (not loop-based) — fits in our aqr pipeline
- Pairs with existing TInvestAdapter: load_prices → vectorbt.run()
- One known caveat: project maintenance slowed (last release Mar 2025);
  we acknowledge it in AGENTS.md.

Usage:
    from aqr.screener.vectorbt import screen_momentum
    top = screen_momentum("SBER", "2022-01", "2024-12", top_n=10)
    # top = [{"fast": 8, "slow": 50, "sharpe": 2.14, ...}, ...]
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class VariantResult:
    """Результат одного бэктеста из grid search."""

    fast: int
    slow: int
    sharpe: float
    sortino: float
    max_drawdown: float
    total_return: float
    n_trades: int


def _require_vectorbt():
    """Lazy import — vectorbt is heavy (Numba + scikit-learn)."""
    try:
        import vectorbt as vbt
    except ImportError as e:
        raise RuntimeError(
            "vectorbt not installed. Install with: "
            "pip install vectorbt"
        ) from e
    return vbt


def screen_momentum(
    ticker: str,
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
    fast_range: tuple[int, int, int] = (5, 50, 5),
    slow_range: tuple[int, int, int] = (20, 120, 5),
    top_n: int = 10,
    candles: pd.DataFrame | None = None,  # optional pre-loaded prices
) -> list[dict]:
    """Grid-search SMA-crossover momentum на тикере.

    Args:
        ticker: T-Invest тикер (SBER / GAZP / и т.д.)
        start_date / end_date: ISO date strings
        fast_range: (start, stop, step) для fast SMA window
        slow_range: (start, stop, step) для slow SMA window
        top_n: сколько лучших вариантов вернуть (sorted by Sharpe desc)
        candles: предзагруженные OHLCV (если None — загружаем через T-Invest)

    Returns:
        list[dict] — каждый элемент asdict(VariantResult) + ticker
        (для удобства передачи в PipelineResult).
    """
    vbt = _require_vectorbt()
    import numpy as np

    # 1. Load prices — если не переданы, идём в T-Invest
    if candles is None:
        from aqr.data.tinvest import TInvestAdapter
        import asyncio

        adapter = TInvestAdapter()
        candles = asyncio.run(
            adapter.candles(ticker, start_date, end_date, interval="D1")
        )
    close = candles["close"]

    # 2. Build parameter grid (filter: slow > fast + 5 to avoid noise)
    fast_params = list(range(*fast_range))
    slow_params = list(range(*slow_range))
    pairs = [(f, s) for f in fast_params for s in slow_params if s > f + 5]

    if not pairs:
        return []

    # 3. Loop: для каждой (fast, slow) пары отдельный бэктест.
    #    vbt's per-window calls are JIT-cached, so each iteration is fast.
    results: list[VariantResult] = []
    for f, s in pairs:
        fast_ma = vbt.MA.run(close, window=f, short_name="f")
        slow_ma = vbt.MA.run(close, window=s, short_name="s")

        entries = fast_ma.ma_crossed_above(slow_ma)
        exits = fast_ma.ma_crossed_below(slow_ma)

        portfolio = vbt.Portfolio.from_signals(
            close,
            entries=entries,
            exits=exits,
            size=1.0,
            size_type="percent",
            init_cash=100_000,
            freq="1D",
        )

        sh = _safe(portfolio.sharpe_ratio())
        so = _safe(portfolio.sortino_ratio())
        dd = _safe(portfolio.max_drawdown())
        tr = _safe(portfolio.total_return())
        nt = _safe_count(portfolio.trades.count())

        results.append(VariantResult(
            fast=f, slow=s, sharpe=sh, sortino=so, max_drawdown=dd,
            total_return=tr, n_trades=nt,
        ))

    # 4. Sort desc by Sharpe, take top_n
    results.sort(key=lambda r: r.sharpe, reverse=True)
    top = results[:top_n]

    return [{**asdict(r), "ticker": ticker} for r in top]


def _safe(x) -> float:
    """NaN-safe float extraction without forcing numpy import."""
    try:
        if x is None:
            return 0.0
        v = float(x)
        if v != v:  # NaN check
            return 0.0
        return v
    except (TypeError, ValueError):
        return 0.0


def _safe_count(x) -> int:
    """NaN-safe int extraction."""
    try:
        if x is None:
            return 0
        v = int(x)
        if v < 0 or v != v:  # NaN
            return 0
        return v
    except (TypeError, ValueError):
        return 0
