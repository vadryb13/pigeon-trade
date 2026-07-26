"""Тесты TInvestAdapter — FIGI cache, intervals, sandbox target, error propagation.

Mock Client (gRPC), проверяем контракт без реального T-Инвестиций.
"""
from __future__ import annotations

import os
import sys
import types
from datetime import UTC
from unittest.mock import MagicMock

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


class FakeQuotation:
    def __init__(self, units: int, nano: int = 0):
        self.units = units
        self.nano = nano


class FakeCandle:
    def __init__(self, time, open_units, high_units, low_units, close_units, volume):
        self.time = time
        self.open = FakeQuotation(open_units)
        self.high = FakeQuotation(high_units)
        self.low = FakeQuotation(low_units)
        self.close = FakeQuotation(close_units)
        self.volume = volume


class FakeInstrument:
    def __init__(self, figi: str, ticker: str):
        self.figi = figi
        self.ticker = ticker


class FakeInstrumentsResponse:
    def __init__(self, instruments: list):
        self.instruments = instruments


class FakeGetCandlesResponse:
    def __init__(self, candles):
        self.candles = candles


class FakeMarketDataService:
    """Рекордер вызовов get_candles + возврат заранее заданных candles."""

    def __init__(self, candles):
        self._candles = candles
        self.calls = []

    def get_candles(self, *, figi, from_, to, interval):
        self.calls.append({
            "figi": figi, "from": from_, "to": to, "interval": interval,
        })
        return FakeGetCandlesResponse(self._candles)


class FakeInstrumentsService:
    def __init__(self, figi_by_ticker: dict[str, str]):
        self._figi_by_ticker = figi_by_ticker
        self.calls = []

    def get_instrument_by_ticker(self, *, ticker, class_code):
        self.calls.append({"ticker": ticker, "class_code": class_code})
        figi = self._figi_by_ticker.get(ticker)
        if figi is None:
            return FakeInstrumentsResponse([])
        return FakeInstrumentsResponse([FakeInstrument(figi, ticker)])


def _make_fake_tinvest_module(monkeypatch, figi_by_ticker, candles):
    """Создаёт фейковый t_tech.invest модуль с Client + CandleInterval."""
    class FakeCandleInterval:
        CANDLE_INTERVAL_1_MIN = "CANDLE_INTERVAL_1_MIN"
        CANDLE_INTERVAL_5_MIN = "CANDLE_INTERVAL_5_MIN"
        CANDLE_INTERVAL_15_MIN = "CANDLE_INTERVAL_15_MIN"
        CANDLE_INTERVAL_HOUR = "CANDLE_INTERVAL_HOUR"
        CANDLE_INTERVAL_DAY = "CANDLE_INTERVAL_DAY"
        CANDLE_INTERVAL_WEEK = "CANDLE_INTERVAL_WEEK"
        CANDLE_INTERVAL_MONTH = "CANDLE_INTERVAL_MONTH"

    class FakeClient:
        INVEST_GRPC_API = "prod-target"
        INVEST_GRPC_API_SANDBOX = "sandbox-target"

        def __init__(self, token, *, target=None):
            self.token = token
            self.target = target
            self.market_data = FakeMarketDataService(candles)
            self.instruments = FakeInstrumentsService(figi_by_ticker)
            FakeClient._instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

    FakeClient._instances = []

    fake = types.ModuleType("t_tech.invest")
    fake.Client = FakeClient
    fake.CandleInterval = FakeCandleInterval
    fake.INVEST_GRPC_API = "prod-target"
    fake.INVEST_GRPC_API_SANDBOX = "sandbox-target"
    monkeypatch.setitem(sys.modules, "t_tech.invest", fake)
    return fake


@pytest.fixture(autouse=True)
def _clear_figi_cache():
    from aqr.data import tinvest as tinvest_mod
    from aqr.data.tinvest import TInvestAdapter
    TInvestAdapter.clear_figi_cache()
    tinvest_mod._tinvest_module = None  # сброс кэша lazy-import
    yield
    TInvestAdapter.clear_figi_cache()
    tinvest_mod._tinvest_module = None


def _sample_candles():
    """100 daily candles для тестов (>=100 чтобы пройти валидацию)."""
    from datetime import datetime, timedelta, timezone
    base = datetime(2023, 1, 2, tzinfo=UTC)
    return [
        FakeCandle(
            time=base + timedelta(days=i),
            open_units=100 + i, high_units=101 + i,
            low_units=99 + i, close_units=100 + i,
            volume=1_000_000,
        )
        for i in range(120)
    ]


class TestCandlesD1:
    def test_returns_dataframe_with_correct_columns(self, monkeypatch, with_credentials):
        candles = _sample_candles()
        _make_fake_tinvest_module(monkeypatch, {"SBER": "BBG004730N88"}, candles)

        from aqr.data.tinvest import TInvestAdapter

        adapter = TInvestAdapter()
        df = adapter.candles("SBER", "2023-01-02", "2024-12-31", interval="D1")

        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert len(df) == 120
        assert df["close"].iloc[0] == 100.0

    def test_resolves_figi_once_then_caches(self, monkeypatch, with_credentials):
        candles = _sample_candles()
        fake = _make_fake_tinvest_module(
            monkeypatch, {"SBER": "BBG004730N88"}, candles,
        )

        from aqr.data.tinvest import TInvestAdapter

        adapter = TInvestAdapter()
        adapter.candles("SBER", "2023-01-02", "2024-12-31")
        adapter.candles("SBER", "2023-01-02", "2024-12-31")
        adapter.candles("SBER", "2023-06-01", "2024-12-31")  # другой диапазон — кэш работает

        # Все три вызова candles использовали один FIGI из кэша
        instances = fake.Client._instances
        assert sum(
            len(inst.instruments.calls) for inst in instances
        ) == 1, f"FIGI должен резолвиться один раз, got {sum(len(inst.instruments.calls) for inst in instances)}"

    def test_unknown_ticker_raises(self, monkeypatch, with_credentials):
        _make_fake_tinvest_module(monkeypatch, {}, _sample_candles())

        from aqr.data.tinvest import TInvestAdapter

        adapter = TInvestAdapter()
        with pytest.raises(ValueError, match="Unknown ticker: 'FAKE_TICKER'"):
            adapter.candles("FAKE_TICKER", "2023-01-02", "2024-12-31")


class TestIntervalMap:
    def test_all_seven_intervals(self):
        from aqr.data.tinvest import INTERVAL_MAP
        assert set(INTERVAL_MAP.keys()) == {"1m", "5m", "15m", "H1", "D1", "W", "M"}

    def test_invalid_interval_raises(self, monkeypatch, with_credentials):
        _make_fake_tinvest_module(monkeypatch, {"SBER": "BBG004730N88"}, [])

        from aqr.data.tinvest import TInvestAdapter

        adapter = TInvestAdapter()
        with pytest.raises(ValueError, match="Unsupported interval"):
            adapter.candles("SBER", "2023-01-02", "2024-12-31", interval="2H")


class TestSandboxTarget:
    def test_sandbox_default_true(self, monkeypatch, with_credentials):
        """Без env — sandbox=True (дефолт для dev/CI)."""
        monkeypatch.delenv("INVEST_SANDBOX", raising=False)
        _make_fake_tinvest_module(monkeypatch, {"SBER": "BBG004730N88"}, _sample_candles())

        from aqr.data.tinvest import TInvestAdapter

        adapter = TInvestAdapter()
        assert adapter.sandbox is True
        assert adapter._target == "sandbox-target"

    def test_invest_sandbox_0_means_production(self, monkeypatch):
        """Без credentials + env INVEST_SANDBOX=0 → production."""
        monkeypatch.setenv("INVEST_SANDBOX", "0")
        monkeypatch.setenv("INVEST_TOKEN", "t.test")
        _make_fake_tinvest_module(monkeypatch, {"SBER": "BBG004730N88"}, _sample_candles())

        from aqr.data.tinvest import TInvestAdapter

        adapter = TInvestAdapter(token="t.test")  # явно, без ContextVar
        assert adapter.sandbox is False
        assert adapter._target == "prod-target"

    def test_credentials_sandbox_overrides_env(self, monkeypatch):
        """Credentials.invest_sandbox=True (дефолт) — env не имеет значения."""
        monkeypatch.setenv("INVEST_SANDBOX", "0")

        from aqr.agent.context import reset_credentials, set_credentials
        from aqr.registry import DecryptedSettings

        creds = DecryptedSettings(
            session_id="alice",
            llm_model="claude-3-5-sonnet-20241022",
            llm_api_key="sk-ant",
            openai_api_key="sk-oai",
            invest_token="t",
            invest_sandbox=True,  # ← через credentials
        )
        token = set_credentials(creds)
        try:
            _make_fake_tinvest_module(
                monkeypatch, {"SBER": "BBG004730N88"}, _sample_candles(),
            )

            from aqr.data.tinvest import TInvestAdapter

            adapter = TInvestAdapter()
            assert adapter.sandbox is True
        finally:
            reset_credentials(token)


class TestErrorPropagation:
    def test_no_retry_no_circuit_breaker(self, monkeypatch, with_credentials):
        """Сетевая ошибка → raise напрямую, без retry."""
        from aqr.data import tinvest as tinvest_mod

        class _BrokenClient:
            INVEST_GRPC_API = "prod"
            INVEST_GRPC_API_SANDBOX = "sandbox"

            def __init__(self, token, *, target=None):
                pass

            def __enter__(self):
                raise ConnectionError("gRPC unavailable")

            def __exit__(self, *a):
                return None

        # Подменяем t_tech.invest module с broken Client
        fake = types.ModuleType("t_tech.invest")
        fake.Client = _BrokenClient
        fake.CandleInterval = MagicMock()
        fake.INVEST_GRPC_API = "prod"
        fake.INVEST_GRPC_API_SANDBOX = "sandbox"
        monkeypatch.setitem(sys.modules, "t_tech.invest", fake)

        # Сбрасываем lazy-кэш tinvest
        tinvest_mod._tinvest_module = None

        from aqr.data.tinvest import TInvestAdapter

        adapter = TInvestAdapter()
        with pytest.raises(ConnectionError, match="gRPC unavailable"):
            adapter.candles("SBER", "2023-01-02", "2024-12-31")


class TestCredentialsRequired:
    def test_raises_without_credentials(self, monkeypatch):
        """Без api_key и без ContextVar → RuntimeError."""
        from aqr.agent.context import current_credentials

        assert current_credentials() is None

        from aqr.data.tinvest import TInvestAdapter

        with pytest.raises(RuntimeError, match="token not provided"):
            TInvestAdapter()

    def test_uses_credentials_when_no_token_arg(self, monkeypatch, with_credentials):
        """Без token параметром → из ContextVar."""
        _make_fake_tinvest_module(monkeypatch, {"SBER": "BBG004730N88"}, _sample_candles())

        from aqr.data.tinvest import TInvestAdapter

        adapter = TInvestAdapter()  # без token
        df = adapter.candles("SBER", "2023-01-02", "2024-12-31")
        assert len(df) == 120


class TestClearFigiCache:
    def test_clear_figi_cache(self, monkeypatch, with_credentials):
        candles = _sample_candles()
        fake = _make_fake_tinvest_module(
            monkeypatch, {"SBER": "BBG004730N88"}, candles,
        )

        from aqr.data.tinvest import TInvestAdapter

        adapter = TInvestAdapter()
        adapter.candles("SBER", "2023-01-02", "2024-12-31")
        TInvestAdapter.clear_figi_cache()
        adapter.candles("SBER", "2023-01-02", "2024-12-31")

        # После clear — FIGI резолвится заново
        total = sum(len(i.instruments.calls) for i in fake.Client._instances)
        assert total == 2
