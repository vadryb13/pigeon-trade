"""Тесты для OhlcvCache (DuckDB-кэш OHLCV).

Требуют [data] extra (duckdb). Если duckdb не установлен — тесты skip.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import CountingAdapter

duckdb = pytest.importorskip("duckdb")


@pytest.fixture(autouse=True)
def _stable_secret(monkeypatch):
    monkeypatch.setenv("AQR_SESSION_SECRET", "test-secret-padded-to-32-bytes-base64==")


# ── Fixtures ────────────────────────────────────────────────────

@pytest.fixture
def tmp_cache(tmp_path: Path):
    """Свежий кэш во временной директории."""
    from aqr.data.ohlcv_cache import OhlcvCache
    db_path = tmp_path / "test_cache.duckdb"
    yield OhlcvCache(db_path)


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """500 дневных баров GBM, индекс DatetimeIndex."""
    rng = np.random.default_rng(42)
    n = 500
    ret = rng.normal(0.0005, 0.015, n)
    ret[100:200] += 0.002
    close = 100 * np.exp(np.cumsum(ret))
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    df = pd.DataFrame({
        "open": close * (1 + rng.normal(0, 0.003, n)),
        "high": close * (1 + np.abs(rng.normal(0, 0.005, n))),
        "low": close * (1 - np.abs(rng.normal(0, 0.005, n))),
        "close": close,
        "volume": rng.integers(100_000, 1_000_000, n),
        "value": close * rng.integers(100_000, 1_000_000, n),
    }, index=idx)
    df.index.name = "begin"
    return df


# ── Базовые CRUD-операции ────────────────────────────────────────

class TestOhlcvCacheBasics:
    def test_put_get_roundtrip(self, tmp_cache, sample_ohlcv):
        """put 500 баров → get возвращает те же данные."""
        n = tmp_cache.put_cache("SBER", sample_ohlcv, "D")
        assert n == 500

        got = tmp_cache.get_cached("SBER", "2023-01-02", "2025-12-31", "D")
        assert got is not None
        assert len(got) == 500
        # Закрытия должны совпадать (до float precision)
        np.testing.assert_allclose(
            got["close"].values,
            sample_ohlcv["close"].values,
            rtol=1e-9,
        )

    def test_get_miss_returns_none(self, tmp_cache):
        """Несуществующий тикер → None (без строк в кэше)."""
        got = tmp_cache.get_cached("MISSING", "2024-01-01", "2024-12-31", "D")
        assert got is None

    def test_partial_window(self, tmp_cache, sample_ohlcv):
        """put за 2 года → get за 6 месяцев возвращает подмножество."""
        tmp_cache.put_cache("GAZP", sample_ohlcv, "D")

        # Берём январь–июнь 2023 (первые 125 рабочих дней)
        got = tmp_cache.get_cached("GAZP", "2023-01-02", "2023-06-30", "D")
        assert got is not None
        # ~125 рабочих дней в январе–июне
        assert 100 <= len(got) <= 135
        assert got.index.min() >= pd.Timestamp("2023-01-02")
        assert got.index.max() <= pd.Timestamp("2023-06-30")

    def test_invalidate_clears_all(self, tmp_cache, sample_ohlcv):
        """invalidate() без тикера удаляет всё."""
        tmp_cache.put_cache("SBER", sample_ohlcv, "D")
        tmp_cache.put_cache("GAZP", sample_ohlcv, "D")

        n_removed = tmp_cache.invalidate()
        assert n_removed == 1000

        assert tmp_cache.get_cached("SBER", "2023-01-02", "2025-12-31", "D") is None
        assert tmp_cache.get_cached("GAZP", "2023-01-02", "2025-12-31", "D") is None

    def test_invalidate_specific_ticker(self, tmp_cache, sample_ohlcv):
        """invalidate(ticker='SBER') удаляет только этот тикер."""
        tmp_cache.put_cache("SBER", sample_ohlcv, "D")
        tmp_cache.put_cache("GAZP", sample_ohlcv, "D")

        n_removed = tmp_cache.invalidate("SBER")
        assert n_removed == 500

        assert tmp_cache.get_cached("SBER", "2023-01-02", "2025-12-31", "D") is None
        assert tmp_cache.get_cached("GAZP", "2023-01-02", "2025-12-31", "D") is not None

    def test_stats(self, tmp_cache, sample_ohlcv):
        """stats() возвращает количество тикеров и строк."""
        assert tmp_cache.stats() == {"tickers": 0, "rows": 0}

        tmp_cache.put_cache("SBER", sample_ohlcv, "D")
        tmp_cache.put_cache("GAZP", sample_ohlcv, "D")

        st = tmp_cache.stats()
        assert st["tickers"] == 2
        assert st["rows"] == 1000

    def test_different_timeframes_isolated(self, tmp_cache, sample_ohlcv):
        """Тот же тикер, разные таймфреймы — независимые ключи."""
        tmp_cache.put_cache("SBER", sample_ohlcv, "D")
        tmp_cache.put_cache("SBER", sample_ohlcv, "H1")

        # H1 ключ ничего не даёт при запросе "D"-кеша
        # get_cached(SBER, "D") → должен найти
        got_d = tmp_cache.get_cached("SBER", "2023-01-02", "2025-12-31", "D")
        assert got_d is not None
        assert len(got_d) == 500

        # get_cached(SBER, "H1") — тоже находит
        got_h1 = tmp_cache.get_cached("SBER", "2023-01-02", "2025-12-31", "H1")
        assert got_h1 is not None
        assert len(got_h1) == 500

        # Запрос несуществующего таймфрейма → None
        assert tmp_cache.get_cached("SBER", "2023-01-02", "2025-12-31", "M1") is None

    def test_upsert_overwrites_existing(self, tmp_cache, sample_ohlcv):
        """Повторный put для того же (ticker, timeframe, begin) — обновляет строку."""
        df1 = sample_ohlcv.copy()
        df1["close"] = 100.0  # первая версия
        tmp_cache.put_cache("SBER", df1, "D")

        df2 = sample_ohlcv.copy()
        df2["close"] = 200.0  # вторая версия
        tmp_cache.put_cache("SBER", df2, "D")

        got = tmp_cache.get_cached("SBER", "2023-01-02", "2025-12-31", "D")
        assert got is not None
        # После upsert — значения из df2
        assert np.allclose(got["close"].iloc[:5].values, [200.0] * 5)
        # Всего строк ровно 500 (не 1000)
        assert len(got) == 500


# ── Интеграция с load_prices ─────────────────────────────────────

class TestLoadPricesIntegration:
    @pytest.mark.asyncio
    async def test_load_prices_uses_cache_on_second_call(
        self, tmp_path, monkeypatch, with_credentials
    ):
        """Второй вызов load_prices для того же тикера не ходит в T-Invest."""
        from aqr.data import ohlcv_cache as cache_mod

        # Подменяем кэш на временную директорию
        db_path = tmp_path / "test_load_cache.duckdb"

        # Подменяем TInvestAdapter так, чтобы первый вызов шёл в T-Invest,
        # второй — должен быть кэширован (иначе call_count будет 2)
        from aqr.data import tinvest as tinvest_mod

        adapter = CountingAdapter()
        monkeypatch.setattr(tinvest_mod, "TInvestAdapter", lambda *a, **kw: adapter)

        # Подменяем путь к кэшу через monkeypatch
        original_init = cache_mod.OhlcvCache.__init__

        def custom_init(self, db_path_arg="data/ohlcv_cache.duckdb"):
            original_init(self, db_path)

        monkeypatch.setattr(cache_mod.OhlcvCache, "__init__", custom_init)

        from aqr.tools.core import load_prices

        # Первый вызов — идёт в T-Invest, заполняет кэш
        r1 = await load_prices(tickers=["SBER"], start_date="2023-01-02", end_date="2024-12-31")
        assert "SBER" in r1
        assert adapter.call_count == 1

        # Второй вызов — должен использовать кэш (T-Invest не вызывается)
        r2 = await load_prices(tickers=["SBER"], start_date="2023-01-02", end_date="2024-12-31")
        assert "SBER" in r2
        assert adapter.call_count == 1  # всё ещё 1, второй запрос не пошёл в T-Invest

        # Данные идентичны
        assert r1["SBER"] == r2["SBER"]


class TestOhlcvCacheEdgeCases:
    def test_put_skips_nan_close_rows(self, tmp_path):
        """Строки с NaN/None close пропускаются, не валят соединение."""
        from aqr.data.ohlcv_cache import OhlcvCache
        db_path = tmp_path / "nan_test.duckdb"
        cache = OhlcvCache(db_path)

        idx = pd.date_range("2024-01-01", periods=5, freq="B")
        df = pd.DataFrame({
            "open": [100, 101, None, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103],
            "close": [100, float("nan"), 102, None, 104],  # две битые строки
            "volume": [1000, 2000, 3000, 4000, 5000],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        }, index=idx)
        df.index.name = "begin"

        n = cache.put_cache("SBER", df, "D")
        # 5 строк исходных — 2 отброшены → 3 валидных
        assert n == 3

        # Кэш не "сломан" — get_cached работает
        got = cache.get_cached("SBER", "2024-01-01", "2024-12-31", "D")
        assert got is not None
        assert len(got) == 3

    def test_default_path_uses_env_var(self, tmp_path, monkeypatch):
        """AQR_CACHE_DIR переопределяет дефолт."""
        target_dir = tmp_path / "env_test"
        monkeypatch.setenv("AQR_CACHE_DIR", str(target_dir))
        from aqr.data.ohlcv_cache import OhlcvCache
        cache = OhlcvCache()  # без аргументов
        # Путь должен быть <env_dir>/<basename>
        expected = target_dir / OhlcvCache.DEFAULT_BASENAME
        assert cache.db_path == expected
        # Каталог создан
        assert target_dir.exists()


