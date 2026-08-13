"""AnalystAgent: runs VectorBT screener + backtest_one for each hypothesis.

For each (ticker, family) pair:
  - momentum: VectorBT screener → deep backtest top-N
  - Other families: generate hypotheses → deep backtest each

All families also get a default-parameter backtest if nothing else works.
"""

from __future__ import annotations

import asyncio
from typing import Any

from aqr.tools import registry as tool_registry

from .base import AgentResult, BaseAgent

MIN_PRICES = 126
DEFAULT_N_HYPOTHESES = 5


class AnalystAgent(BaseAgent):
    """Analyzes a single ticker across multiple families."""

    name = "analyst"

    async def analyze(
        self,
        ticker: str,
        families: list[str],
        start_date: str = "2023-01-01",
        end_date: str = "2024-12-31",
        timeframe: str = "D1",
        top_n: int = 5,
    ) -> AgentResult:
        """Run screening + deep backtest for one ticker across families.

        Returns:
            {"ticker", "results": [BacktestResult-dict, ...],
             "n_screened", "n_deep"}
        """
        self.logger.info("analyzing %s families=%s", ticker, families)

        # Phase 1: Load prices once
        load_tool = tool_registry.get("load_prices")
        if load_tool is None:
            return self._fail("load_prices tool not registered")

        try:
            prices_raw = await load_tool.fn(
                tickers=[ticker],
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe,
            )
        except Exception as exc:
            return self._fail(f"load_prices failed for {ticker}: {exc}")

        prices_list = prices_raw.get(ticker, [])
        if len(prices_list) < MIN_PRICES:
            return self._fail(
                f"{ticker}: insufficient data ({len(prices_list)} < {MIN_PRICES})"
            )

        # Phase 2: For each family, produce hypothesis specs then deep-backtest
        all_results: list[dict] = []
        n_screened = 0

        for family in families:
            specs = await self._build_specs(ticker, family, start_date, end_date)
            n_screened += len(specs)

            specs_to_test = specs[:top_n] if top_n is not None else specs
            for spec in specs_to_test:
                deep = await self._deep_backtest(
                    ticker=ticker,
                    family=family,
                    params=spec,
                    prices=prices_list,
                )
                if deep is not None:
                    all_results.append(deep)

        return self._ok(
            ticker=ticker,
            results=all_results,
            n_screened=n_screened,
            n_deep=len(all_results),
        )

    async def _build_specs(
        self,
        ticker: str,
        family: str,
        start_date: str,
        end_date: str,
    ) -> list[dict[str, Any]]:
        """Build parameter specs for a (ticker, family) pair.

        For momentum: run VectorBT screener for pareto-optimal params.
        For others: use generate_hypotheses tool.
        Fallback: return a single default-params spec.
        """
        specs: list[dict] = []

        # --- Screener path (momentum only) ---
        if family == "momentum":
            screen = await self._screening(ticker, start_date, end_date)
            if screen:
                return [{"fast": s["fast"], "slow": s["slow"]} for s in screen]

        # --- Tool-generated hypotheses ---
        gen_tool = tool_registry.get("generate_hypotheses")
        if gen_tool is not None:
            try:
                hyps = await gen_tool.fn(
                    tickers=[ticker],
                    families=[family],
                    n=DEFAULT_N_HYPOTHESES,
                )
                specs = [
                    dict(h.get("params", {}))
                    for h in hyps
                    if h.get("params")
                ]
            except Exception:
                self.logger.exception(
                    "generate_hypotheses failed for %s/%s", ticker, family,
                )
                raise

        # --- Fallback: one default-param spec ---
        if not specs:
            specs = await self._default_specs(family)

        return specs

    async def _screening(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> list[dict]:
        """Run VectorBT screener for momentum parameters.

        VectorBT — Numba-ускоренный sync-пайплайн. Запускаем в thread-pool,
        чтобы не блокировать event loop (B10).
        """
        try:
            from aqr.screener.vectorbt import screen_momentum
        except ImportError as exc:
            raise RuntimeError("momentum screener dependency is not installed") from exc

        try:
            return await asyncio.to_thread(
                screen_momentum,
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                top_n=20,
            )
        except Exception as exc:
            self.logger.exception("screener failed for %s", ticker)
            raise RuntimeError(f"screener failed for {ticker}") from exc

    async def _default_specs(self, family: str) -> list[dict[str, Any]]:
        """Return one default-param spec for a family.

        Used when no other spec source is available.
        """
        defaults: dict[str, list[dict]] = {
            "momentum": [{"fast": 10, "slow": 30}],
            "mean_reversion": [{"window": 20, "entry_z": -2.0, "exit_z": 0.5}],
            "breakout": [{"window": 20, "multiplier": 2.0}],
            "volatility": [{"window": 20, "lookback": 60}],
        }
        return defaults.get(family, [{}])

    async def _deep_backtest(
        self,
        ticker: str,
        family: str,
        params: dict[str, Any],
        prices: list[float],
    ) -> dict[str, Any] | None:
        """Run backtest_one on a single parameter combo."""
        bt_tool = tool_registry.get("backtest_one")
        if bt_tool is None:
            raise RuntimeError("backtest_one tool not registered")

        hypothesis: dict[str, Any] = {
            "name": f"{family}_{ticker}",
            "family": family,
            "ticker": ticker,
            "params": params,
        }

        try:
            result = await bt_tool.fn(hypothesis=hypothesis, prices=prices)
            if "error" in result:
                self.logger.warning(
                    "backtest_one error: %s/%s %s → %s",
                    ticker, family, params, result["error"],
                )
                return None
            return result
        except Exception as exc:
            self.logger.exception(
                "backtest_one exception: %s/%s %s", ticker, family, params,
            )
            raise RuntimeError(f"backtest failed for {ticker}/{family}") from exc
