"""Тесты: агент переиспользует DuckDB-кэш для повторных прогонов.

Автоматизирует ручной тест из TASKS.md 7.4:
    "проверь momentum на Сбере" → "а что если mean reversion?" →
    агент переиспользует кэш и не ходит в MOEX повторно.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def moex_call_counter(monkeypatch):
    """Подменяем MOEXAdapter.candles — считаем вызовы и возвращаем синтетику."""
    from aqr.data import moex as moex_mod
    from aqr.data import ohlcv_cache as cache_mod

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
                "value": np.zeros(n),
            }, index=idx)

    monkeypatch.setattr(moex_mod, "MOEXAdapter", _CountingAdapter)

    # Подменяем путь к кэшу на временную директорию
    import os
    db_path = Path(os.getenv("AQR_CACHE_TEST_DIR", "/tmp/aqr_cache_test.duckdb"))

    original_init = cache_mod.OhlcvCache.__init__

    def custom_init(self, db_path_arg="data/ohlcv_cache.duckdb"):
        original_init(self, db_path)

    monkeypatch.setattr(cache_mod.OhlcvCache, "__init__", custom_init)

    # Очищаем кэш перед тестом
    if db_path.exists():
        db_path.unlink()

    yield counter

    # Cleanup
    if db_path.exists():
        db_path.unlink()


class TestCacheReuseAcrossRuns:
    @pytest.mark.asyncio
    async def test_second_load_prices_does_not_hit_moex(self, moex_call_counter):
        """Два вызова load_prices для того же тикера → MOEX вызван 1 раз."""
        from aqr.tools.core import load_prices

        # Первый вызов — кэш пуст, идём в MOEX
        r1 = await load_prices(["SBER"], "2023-01-02", "2024-12-31")
        assert "SBER" in r1
        assert moex_call_counter["n"] == 1

        # Второй вызов — данные в кэше, MOEX не дёргается
        r2 = await load_prices(["SBER"], "2023-01-02", "2024-12-31")
        assert r1["SBER"] == r2["SBER"]
        assert moex_call_counter["n"] == 1, "MOEX был вызван повторно — кэш не сработал"

    @pytest.mark.asyncio
    async def test_followup_question_reuses_cache(self, moex_call_counter):
        """TASKS.md 7.4: 'momentum → mean reversion' на Сбере переиспользует кэш."""
        from aqr.tools.core import load_prices

        # 1. Первое обращение — кэш пуст
        await load_prices(["SBER"], "2023-01-02", "2024-12-31")
        first_call_count = moex_call_counter["n"]
        assert first_call_count == 1

        # 2. Пользователь спрашивает про mean reversion (другая семья, тот же тикер)
        # load_prices для SBER — данные уже в кэше
        r = await load_prices(["SBER"], "2023-01-02", "2024-12-31")
        assert "SBER" in r
        # MOEX НЕ вызывается повторно
        assert moex_call_counter["n"] == first_call_count

    @pytest.mark.asyncio
    async def test_new_ticker_still_hits_moex(self, moex_call_counter):
        """Новый тикер → идём в MOEX (кэш пуст для него)."""
        from aqr.tools.core import load_prices

        await load_prices(["SBER"], "2023-01-02", "2024-12-31")
        assert moex_call_counter["n"] == 1

        # Новый тикер — промах в кэше → MOEX
        await load_prices(["GAZP"], "2023-01-02", "2024-12-31")
        assert moex_call_counter["n"] == 2

        # Старый тикер — из кэша
        await load_prices(["SBER"], "2023-01-02", "2024-12-31")
        assert moex_call_counter["n"] == 2

    @pytest.mark.asyncio
    async def test_mixed_tickers_partial_cache(self, moex_call_counter):
        """SBER в кэше, GAZP — нет → 1 вызов MOEX для GAZP."""
        from aqr.tools.core import load_prices

        # Warm-up SBER
        await load_prices(["SBER"], "2023-01-02", "2024-12-31")
        assert moex_call_counter["n"] == 1

        # Смешанный запрос: SBER из кэша, GAZP из MOEX
        result = await load_prices(["SBER", "GAZP"], "2023-01-02", "2024-12-31")
        assert set(result.keys()) == {"SBER", "GAZP"}
        # Только GAZP вызвал MOEX
        assert moex_call_counter["n"] == 2

    @pytest.mark.asyncio
    async def test_data_persists_across_adapter_instances(self, moex_call_counter):
        """Кэш живёт между инстансами OhlcvCache (на диске)."""
        from aqr.tools.core import load_prices

        # Прогон 1: warm cache
        await load_prices(["SBER"], "2023-01-02", "2024-12-31")
        first_n = moex_call_counter["n"]

        # Новый инстанс кэша в той же директории
        from aqr.data import ohlcv_cache as cache_mod
        cache_mod.OhlcvCache()  # инициализация из того же пути

        # Прогон 2: cache должен быть валидным
        r = await load_prices(["SBER"], "2023-01-02", "2024-12-31")
        assert moex_call_counter["n"] == first_n
        assert len(r["SBER"]) == 500
