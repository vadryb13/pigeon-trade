"""Unit tests for hypothesis signal functions in aqr.pipeline.hypotheses."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aqr.pipeline.hypotheses import (
    _breakout,
    _mean_reversion,
    _momentum_zscore,
    _sma_crossover,
    _volatility_filter,
    generate_hypotheses,
    make_one_with_params,
)


def _arange_prices(n: int = 200) -> pd.Series:
    """Simple range 100..200 to produce meaningful signals."""
    return pd.Series(
        np.arange(100, 100 + n, dtype=float),
        index=pd.date_range("2023-01-01", periods=n, freq="B"),
    )


def _constant_prices(n: int = 200) -> pd.Series:
    """Constant prices — should result in zero signal (no edge)."""
    return pd.Series(
        [100.0] * n,
        index=pd.date_range("2023-01-01", periods=n, freq="B"),
    )


class TestSMACrossover:
    def test_returns_series_with_correct_index(self):
        prices = _arange_prices()
        sig_fn = _sma_crossover(5, 20)
        result = sig_fn(prices)

        assert isinstance(result, pd.Series)
        assert len(result) == len(prices)
        assert result.index.equals(prices.index)

    def test_values_in_valid_range(self):
        prices = _arange_prices()
        sig_fn = _sma_crossover(5, 20)
        result = sig_fn(prices)

        assert set(result.dropna().unique()).issubset({-1.0, 0.0, 1.0})

    def test_first_slow_points_are_zero(self):
        prices = _arange_prices()
        sig_fn = _sma_crossover(5, 20)
        result = sig_fn(prices)

        assert (result.iloc[:20] == 0.0).all()

    def test_with_identical_prices(self):
        """Constant prices — SMA lines are equal, no crossover → all zero or near-zero."""
        prices = _constant_prices()
        sig_fn = _sma_crossover(5, 20)
        result = sig_fn(prices)

        assert result.notna().all()


class TestMomentumZscore:
    def test_returns_series_with_correct_index(self):
        prices = _arange_prices()
        sig_fn = _momentum_zscore(20, 1.0)
        result = sig_fn(prices)

        assert isinstance(result, pd.Series)
        assert len(result) == len(prices)

    def test_values_in_valid_range(self):
        prices = _arange_prices()
        sig_fn = _momentum_zscore(20, 1.0)
        result = sig_fn(prices)

        valid = set(result.dropna().unique())
        for v in valid:
            assert v in {-1.0, 0.0, 1.0}, f"Unexpected value {v}"

    def test_first_lookback_points_are_zero(self):
        prices = _arange_prices()
        sig_fn = _momentum_zscore(30, 1.0)
        result = sig_fn(prices)

        assert (result.iloc[:30] == 0.0).all()


class TestMeanReversion:
    def test_returns_series(self):
        prices = _arange_prices()
        sig_fn = _mean_reversion(20, 1.0)
        result = sig_fn(prices)

        assert isinstance(result, pd.Series)
        assert len(result) == len(prices)


class TestBreakout:
    def test_returns_series_with_correct_index(self):
        prices = _arange_prices()
        sig_fn = _breakout(20)
        result = sig_fn(prices)

        assert isinstance(result, pd.Series)
        assert len(result) == len(prices)

    def test_values_in_valid_range(self):
        prices = _arange_prices()
        sig_fn = _breakout(20)
        result = sig_fn(prices)

        assert set(result.dropna().unique()).issubset({-1.0, 0.0, 1.0})


class TestVolatilityFilter:
    def test_returns_series(self):
        prices = _arange_prices(200)
        sig_fn = _volatility_filter(20, 0.01)
        result = sig_fn(prices)

        assert isinstance(result, pd.Series)
        assert len(result) == len(prices)

    def test_acts_like_momentum_when_vol_high_enough(self):
        """With very low threshold, vol filter behaves like momentum z-score."""
        prices = _arange_prices(200)
        mom = _momentum_zscore(20, 1.0)(prices)
        vol = _volatility_filter(20, 0.0)(prices)

        pd.testing.assert_series_equal(vol, mom)


class TestGenerateHypotheses:
    def test_returns_n_specs(self):
        specs = generate_hypotheses(
            tickers=["SBER"], families=["momentum"], n=5, seed=42,
        )
        assert len(specs) == 5

    def test_all_have_known_family(self):
        specs = generate_hypotheses(
            tickers=["SBER", "GAZP"],
            families=["momentum", "mean_reversion"],
            n=10, seed=42,
        )
        known = {"momentum", "mean_reversion"}
        for s in specs:
            assert s.family in known

    def test_empty_tickers_returns_empty(self):
        specs = generate_hypotheses(
            tickers=[], families=["momentum"], n=10,
        )
        assert specs == []


class TestMakeOneWithParams:
    def test_known_family_returns_spec(self):
        spec = make_one_with_params("momentum", "SBER", {"fast": 5, "slow": 20})
        assert spec is not None
        assert spec.family == "momentum"
        assert spec.ticker == "SBER"
        assert spec.params == {"fast": 5, "slow": 20}
        assert callable(spec.fn)

    def test_unknown_family_returns_none(self):
        spec = make_one_with_params("nonexistent", "SBER", {})
        assert spec is None

    def test_spec_fn_returns_valid_position(self):
        spec = make_one_with_params("momentum", "SBER", {"fast": 5, "slow": 20})
        prices = _arange_prices()
        pos = spec.fn(prices)

        assert isinstance(pos, pd.Series)
        assert len(pos) == len(prices)
