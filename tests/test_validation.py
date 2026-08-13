"""Statistical validation module tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from aqr.validation import (
    CombinatorialPurgedCV,
    bootstrap_sharpe_ci,
    deflated_sharpe_ratio,
    min_track_record_length,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    whites_reality_check,
)


class TestDeflatedSharpe:
    def test_deflates_with_many_trials(self):
        """DSR should be lower than PSR when n_trials is large."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.01, 252)
        psr = probabilistic_sharpe_ratio(returns)
        dsr_1 = deflated_sharpe_ratio(returns, n_trials=1)
        dsr_1000 = deflated_sharpe_ratio(returns, n_trials=1000)
        assert dsr_1000["deflated_sharpe"] < dsr_1["deflated_sharpe"]
        assert dsr_1["deflated_sharpe"] <= psr + 1e-9

    def test_verdict_significant_only_for_strong_edge(self):
        rng = np.random.default_rng(42)
        # Strong edge: Sharpe ~ 4.5
        strong = rng.normal(0.003, 0.01, 500)
        result = deflated_sharpe_ratio(strong, n_trials=100)
        assert result["deflated_sharpe"] > 0.5

        # Pure noise
        noise = rng.normal(0.0, 0.01, 500)
        result = deflated_sharpe_ratio(noise, n_trials=100)
        assert result["verdict"] == "not_significant"

    def test_insufficient_data(self):
        result = deflated_sharpe_ratio(np.array([0.01, -0.02]), n_trials=10)
        assert result["verdict"] == "insufficient_data"


class TestPBO:
    def test_pbo_high_on_pure_noise(self):
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.01, (500, 50))
        result = probability_of_backtest_overfitting(noise, n_partitions=8, max_combinations=200)
        # Pure noise with 50 candidates should yield PBO clearly > 0.3 (not robust)
        assert result["pbo"] > 0.3
        assert result["verdict"] in ("suspicious", "overfit")

    def test_pbo_low_with_real_signal(self):
        rng = np.random.default_rng(42)
        signal = rng.normal(0.001, 0.008, (500, 1))
        noise = rng.normal(0, 0.01, (500, 49))
        mixed = np.hstack([signal, noise])
        result = probability_of_backtest_overfitting(mixed, n_partitions=8, max_combinations=200)
        assert result["pbo"] < 0.5


class TestCPCV:
    def test_returns_expected_n_paths(self):
        ts = pd.date_range("2020-01-01", periods=500, freq="D")
        cv = CombinatorialPurgedCV(n_splits=6, n_test_splits=2)
        assert cv.n_paths() == 15
        splits = list(cv.split(ts))
        assert len(splits) == 15

    def test_no_overlap(self):
        ts = pd.date_range("2020-01-01", periods=500, freq="D")
        cv = CombinatorialPurgedCV(n_splits=5, n_test_splits=1, embargo_pct=0.0)
        for train, test in cv.split(ts):
            assert len(set(train) & set(test)) == 0

    def test_embargo_applied(self):
        ts = pd.date_range("2020-01-01", periods=500, freq="D")
        cv_no_emb = CombinatorialPurgedCV(n_splits=5, n_test_splits=1, embargo_pct=0.0)
        cv_emb = CombinatorialPurgedCV(n_splits=5, n_test_splits=1, embargo_pct=0.05)
        splits_no = list(cv_no_emb.split(ts))
        splits_emb = list(cv_emb.split(ts))
        # With embargo, train sets should be smaller
        for (t1, _), (t2, _) in zip(splits_no, splits_emb):
            assert len(t2) <= len(t1)


class TestRealityCheck:
    def test_reality_check_runs(self):
        rng = np.random.default_rng(42)
        strategies = rng.normal(0, 0.01, (250, 5))
        benchmark = rng.normal(0, 0.01, 250)
        result = whites_reality_check(strategies, benchmark, n_bootstrap=200)
        assert "p_value" in result
        assert 0 <= result["p_value"] <= 1
        assert result["best_strategy_idx"] in range(5)

    def test_bootstrap_ci(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.01, 500)
        ci = bootstrap_sharpe_ci(returns, n_bootstrap=200)
        assert ci["ci_low"] < ci["sharpe"] < ci["ci_high"]
        assert 0 <= ci["p_positive"] <= 1


class TestMinTRL:
    def test_returns_positive_for_beatable_target(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.002, 0.01, 500)  # Sharpe ~ 3
        trl = min_track_record_length(returns, sr_target=1.0)
        assert trl > 0

    def test_returns_negative_for_unbeatable(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.01, 500)
        trl = min_track_record_length(returns, sr_target=2.0)
        assert trl == -1
