"""Тесты: load_prices переиспользует DuckDB-кэш для повторных прогонов.

Автоматизирует ручной тест из TASKS.md 7.4:
    "проверь momentum на Сбере" → "а что если mean reversion?" →
    агент переиспользует кэш и не ходит в T-Invest повторно.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _stable_secret(monkeypatch):
    monkeypatch.setenv("AQR_SESSION_SECRET", "test-secret-padded-to-32-bytes-base64==")


@pytest.fixture
def with_credentials():
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


@pytest.fixture
def tinvest_counter(monkeypatch):
    """Подменяем TInvestAdapter.candles — считаем вызовы и возвращаем синтетику."""
    from aqr.data import ohlcv_cache as cache_mod
    from aqr.data import tinvest as tinvest_mod

    counter = {"n": 0}

    class _CountingAdapter:
        def __init__(self, *a, **kw):
            pass

        def candles(self, ticker, *a, **kw):
            counter["n"] += 1
            rng = np.random.default_rng(hash(ticker) % (2**32))
            n = 500
            ret = rng.normal(0.0005, 0.015, n)
            idx = pd.date_range("2023-01-02", periods=n, freq="B")
            px = 100 * np.exp(np.cumsum(ret))
            return pd.DataFrame({
                "open": px, "high": px * 1.001, "low": px * 0.999,
                "close": px, "volume": np.zeros(n, dtype=np.int64),
            }, index=idx)

    monkeypatch.setattr(tinvest_mod, "TInvestAdapter", _CountingAdapter)

    # Подменяем путь к кэшу на временную директорию
    db_path = Path(os.getenv("AQR_CACHE_TEST_DIR", "/tmp/aqr_cache_test.duckdb"))

    original_init = cache_mod.OhlcvCache.__init__

    def custom_init(self, db_path_arg="data/ohlcv_cache.duckdb"):
        original_init(self, db_path)

    monkeypatch.setattr(cache_mod.OhlcvCache, "__init__", custom_init)

    # Очищаем кэш перед тестом
    if db_path.exists():
        db_path.unlink()

    yield counter

    if db_path.exists():
        db_path.unlink()


class TestCacheReuseAcrossRuns:
    @pytest.mark.asyncio
    async def test_second_load_prices_does_not_hit_tinvest(
        self, tinvest_counter, with_credentials
    ):
        """Два вызова load_prices для того же тикера → T-Invest вызван 1 раз."""
        from aqr.tools.core import load_prices

        r1 = await load_prices(["SBER"], "2023-01-02", "2024-12-31")
        assert "SBER" in r1
        assert tinvest_counter["n"] == 1

        r2 = await load_prices(["SBER"], "2023-01-02", "2024-12-31")
        assert r1["SBER"] == r2["SBER"]
        assert tinvest_counter["n"] == 1, "T-Invest вызван повторно — кэш не сработал"

    @pytest.mark.asyncio
    async def test_followup_question_reuses_cache(
        self, tinvest_counter, with_credentials
    ):
        """TASKS.md 7.4: 'momentum → mean reversion' на Сбере переиспользует кэш."""
        from aqr.tools.core import load_prices

        await load_prices(["SBER"], "2023-01-02", "2024-12-31")
        first_call_count = tinvest_counter["n"]
        assert first_call_count == 1

        r = await load_prices(["SBER"], "2023-01-02", "2024-12-31")
        assert "SBER" in r
        assert tinvest_counter["n"] == first_call_count

    @pytest.mark.asyncio
    async def test_new_ticker_still_hits_tinvest(
        self, tinvest_counter, with_credentials
    ):
        """Новый тикер → идём в T-Invest (кэш пуст для него)."""
        from aqr.tools.core import load_prices

        await load_prices(["SBER"], "2023-01-02", "2024-12-31")
        assert tinvest_counter["n"] == 1

        await load_prices(["GAZP"], "2023-01-02", "2024-12-31")
        assert tinvest_counter["n"] == 2

        await load_prices(["SBER"], "2023-01-02", "2024-12-31")
        assert tinvest_counter["n"] == 2

    @pytest.mark.asyncio
    async def test_mixed_tickers_partial_cache(
        self, tinvest_counter, with_credentials
    ):
        """SBER в кэше, GAZP — нет → 1 вызов T-Invest для GAZP."""
        from aqr.tools.core import load_prices

        await load_prices(["SBER"], "2023-01-02", "2024-12-31")
        assert tinvest_counter["n"] == 1

        result = await load_prices(["SBER", "GAZP"], "2023-01-02", "2024-12-31")
        assert set(result.keys()) == {"SBER", "GAZP"}
        assert tinvest_counter["n"] == 2

    @pytest.mark.asyncio
    async def test_data_persists_across_cache_instances(
        self, tinvest_counter, with_credentials
    ):
        """Кэш живёт между инстансами OhlcvCache (на диске)."""
        from aqr.tools.core import load_prices

        await load_prices(["SBER"], "2023-01-02", "2024-12-31")
        first_n = tinvest_counter["n"]

        from aqr.data import ohlcv_cache as cache_mod
        cache_mod.OhlcvCache()  # инициализация из того же пути

        r = await load_prices(["SBER"], "2023-01-02", "2024-12-31")
        assert tinvest_counter["n"] == first_n
        assert len(r["SBER"]) == 500
