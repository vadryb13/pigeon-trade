"""Тесты для aqr.screener.vectorbt — fast grid-search momentum strategies.

VectorBT optional dep. Тесты skip если не установлен.
"""
from __future__ import annotations

import pandas as pd
import pytest

# Skip весь файл если vectorbt не установлен
vbt = pytest.importorskip("vectorbt")


@pytest.fixture
def synthetic_prices():
    """Синтетический ряд для теста — без сети/БД."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    n = 500
    drift = 0.0005
    vol = 0.015
    rets = rng.normal(drift, vol, n)
    rets[100:200] += 0.002  # бычий тренд
    rets[300:400] -= 0.001
    px = 100 * np.exp(np.cumsum(rets))
    return pd.Series(px, index=pd.date_range("2023-01-02", periods=n, freq="B"))


class TestVectorBTScreener:
    def test_import_succeeds(self):
        """Модуль импортируется без ошибок."""
        from aqr.screener import VariantResult, screen_momentum
        assert callable(screen_momentum)
        assert VariantResult is not None

    def test_screen_momentum_returns_sorted_results(self, synthetic_prices):
        """screen_momentum возвращает список отсортированный по Sharpe desc."""
        from aqr.screener import screen_momentum

        result = screen_momentum(
            "SBER", "2023-01-02", "2024-12-30",
            fast_range=(5, 15, 5),  # 2 варианта
            slow_range=(20, 50, 10),  # 3 варианта
            top_n=5,
            candles=pd.DataFrame(
                {"close": synthetic_prices, "open": synthetic_prices,
                 "high": synthetic_prices, "low": synthetic_prices,
                 "volume": [0] * len(synthetic_prices)},
                index=synthetic_prices.index,
            ),
        )

        assert isinstance(result, list)
        assert len(result) > 0
        assert len(result) <= 5

        # Каждый результат содержит expected fields
        first = result[0]
        for key in ("ticker", "fast", "slow", "sharpe", "sortino",
                    "max_drawdown", "total_return", "n_trades"):
            assert key in first, f"Missing key: {key}"

        # Sorted desc по Sharpe
        sharpes = [r["sharpe"] for r in result]
        assert sharpes == sorted(sharpes, reverse=True), \
            f"Results not sorted: {sharpes}"

    def test_top_n_limit_respected(self, synthetic_prices):
        """top_n действительно ограничивает размер выдачи."""
        from aqr.screener import screen_momentum

        result = screen_momentum(
            "SBER", "2023-01-02", "2024-12-30",
            fast_range=(5, 30, 5),  # 5
            slow_range=(20, 80, 15),  # 5
            top_n=3,
            candles=pd.DataFrame({"close": synthetic_prices}, index=synthetic_prices.index),
        )
        assert len(result) == 3

    def test_fast_less_than_slow_constraint(self, synthetic_prices):
        """Constraint: fast + 5 < slow (отсекает бессмысленные)."""
        from aqr.screener import screen_momentum

        result = screen_momentum(
            "SBER", "2023-01-02", "2024-12-30",
            fast_range=(3, 20, 1),  # 17 fast values
            slow_range=(20, 100, 5),  # 16 slow values
            top_n=100,
            candles=pd.DataFrame({"close": synthetic_prices}, index=synthetic_prices.index),
        )
        for r in result:
            assert r["slow"] > r["fast"] + 5, \
                f"slow={r['slow']} not > fast+5 ({r['fast']})"

    def test_returns_raises_without_credentials(self, monkeypatch):
        """Без credentials в ContextVar → RuntimeError."""
        from aqr.graph.context import _active_credentials

        # Сбрасываем ContextVar напрямую (reset_credentials ожидает Token, не None)
        saved_token = _active_credentials.set(None)
        _active_credentials.reset(saved_token)
        # После reset() значение будет default (None)
        assert _active_credentials.get() is None
        monkeypatch.delenv("INVEST_TOKEN", raising=False)

        from aqr.screener import screen_momentum
        with pytest.raises(RuntimeError, match="credentials"):
            screen_momentum("SBER", "2023-01-02", "2024-12-30")

    def test_works_with_context_credentials(self, synthetic_prices):
        """Credentials в ContextVar → screen_momentum работает."""
        from aqr.graph.context import reset_credentials, set_credentials
        from aqr.registry import DecryptedSettings
        from aqr.screener import screen_momentum

        creds = DecryptedSettings(
            session_id="alice",
            llm_model="claude-3-5-sonnet-20241022",
            llm_api_key="sk-ant-fake",
            openai_api_key="sk-oai-fake",
            invest_token="t.INVEST_TOKEN_fake",
            invest_sandbox=True,
        )
        token = set_credentials(creds)
        try:
            result = screen_momentum(
                "SBER", "2023-01-02", "2024-12-30",
                fast_range=(5, 10, 5), slow_range=(20, 30, 10), top_n=2,
                candles=pd.DataFrame({"close": synthetic_prices}, index=synthetic_prices.index),
            )
            assert len(result) >= 1
        finally:
            reset_credentials(token)
