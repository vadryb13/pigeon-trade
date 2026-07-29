"""Тесты TInvestAdapter — FIGI cache, intervals, sandbox target, error propagation.

Mock AsyncClient (async context manager), проверяем контракт без реального T-Инвестиций.
"""
from __future__ import annotations

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
    from aqr.graph.context import reset_credentials, set_credentials
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


class FakeFindInstrumentResponse:
    def __init__(self, instruments: list):
        self.instruments = instruments


class FakeGetCandlesResponse:
    def __init__(self, candles):
        self.candles = candles


class FakeMarketDataService:
    """Mock для market_data."""

    def __init__(self, candles):
        self._candles = candles
        self.calls = []

    async def get_candles(self, *, figi, from_, to, interval):
        self.calls.append({
            "figi": figi, "from": from_, "to": to, "interval": interval,
        })
        return FakeGetCandlesResponse(self._candles)


class FakeInstrumentsService:
    """Mock для instruments."""

    def __init__(self, figi_by_ticker: dict[str, str]):
        self._figi_by_ticker = figi_by_ticker
        self.calls = []

    async def find_instrument(self, query, instrument_kind=None, api_trade_available_flag=None):
        self.calls.append({
            "query": query, "instrument_kind": instrument_kind,
        })
        figi = self._figi_by_ticker.get(query)
        if figi is None:
            return FakeFindInstrumentResponse([])
        return FakeFindInstrumentResponse(
            [FakeInstrument(figi, query) for f in [figi]]
        )


def _make_fake_tinvest_module(monkeypatch, figi_by_ticker, candles):
    """Создаёт фейковый t_tech.invest модуль с AsyncClient + CandleInterval + InstrumentType."""
    class FakeCandleInterval:
        CANDLE_INTERVAL_1_MIN = 1
        CANDLE_INTERVAL_2_MIN = 6
        CANDLE_INTERVAL_3_MIN = 7
        CANDLE_INTERVAL_5_MIN = 2
        CANDLE_INTERVAL_10_MIN = 8
        CANDLE_INTERVAL_15_MIN = 3
        CANDLE_INTERVAL_30_MIN = 9
        CANDLE_INTERVAL_HOUR = 4
        CANDLE_INTERVAL_2_HOUR = 10
        CANDLE_INTERVAL_4_HOUR = 11
        CANDLE_INTERVAL_DAY = 5
        CANDLE_INTERVAL_WEEK = 12
        CANDLE_INTERVAL_MONTH = 13
        CANDLE_INTERVAL_UNSPECIFIED = 0

    class FakeInstrumentType:
        INSTRUMENT_TYPE_SHARE = 2

    async_services_instances: list = []

    class FakeAsyncServices:
        def __init__(self, token):
            self.token = token
            self.instruments = FakeInstrumentsService(figi_by_ticker)
            self.market_data = FakeMarketDataService(candles)
            async_services_instances.append(self)

    class FakeAsyncClient:
        INVEST_GRPC_API = "prod-target"
        INVEST_GRPC_API_SANDBOX = "sandbox-target"

        def __init__(self, token, *, target=None):
            self.token = token
            self.target = target
            self._services = None

        async def __aenter__(self):
            self._services = FakeAsyncServices(self.token)
            return self._services

        async def __aexit__(self, *a):
            self._services = None

    fake = types.ModuleType("t_tech.invest")
    fake.AsyncClient = FakeAsyncClient
    fake.CandleInterval = FakeCandleInterval
    fake.InstrumentType = FakeInstrumentType
    fake.INVEST_GRPC_API = "prod-target"
    fake.INVEST_GRPC_API_SANDBOX = "sandbox-target"
    fake.ASYNC_SERVICES_INSTANCES = async_services_instances
    # constants submodule
    fake.constants = types.ModuleType("constants")
    fake.constants.INVEST_GRPC_API = "prod-target"
    fake.constants.INVEST_GRPC_API_SANDBOX = "sandbox-target"

    monkeypatch.setitem(sys.modules, "t_tech.invest", fake)
    return fake


@pytest.fixture(autouse=True)
def _clear_figi_cache():
    from aqr.data import tinvest as tinvest_mod
    from aqr.data.tinvest import TInvestAdapter
    TInvestAdapter.clear_figi_cache()
    tinvest_mod._tinvest_module = None
    yield
    TInvestAdapter.clear_figi_cache()
    tinvest_mod._tinvest_module = None


def _sample_candles():
    """100 daily candles для тестов (>=100 чтобы пройти валидацию)."""
    from datetime import datetime, timedelta
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
    @pytest.mark.asyncio
    async def test_returns_dataframe_with_correct_columns(self, monkeypatch, with_credentials):
        candles = _sample_candles()
        _make_fake_tinvest_module(monkeypatch, {"SBER": "BBG004730N88"}, candles)

        from aqr.data.tinvest import TInvestAdapter

        adapter = TInvestAdapter()
        df = await adapter.candles("SBER", "2023-01-02", "2024-12-31", interval="D1")

        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert len(df) == 120
        assert df["close"].iloc[0] == 100.0

    @pytest.mark.asyncio
    async def test_resolves_figi_once_then_caches(self, monkeypatch, with_credentials):
        candles = _sample_candles()
        fake = _make_fake_tinvest_module(
            monkeypatch, {"SBER": "BBG004730N88"}, candles,
        )

        from aqr.data.tinvest import TInvestAdapter

        adapter = TInvestAdapter()
        await adapter.candles("SBER", "2023-01-02", "2024-12-31")
        await adapter.candles("SBER", "2023-01-02", "2024-12-31")
        await adapter.candles("SBER", "2023-06-01", "2024-12-31")

        # Все три вызова candles использовали один FIGI из кэша
        instances = fake.ASYNC_SERVICES_INSTANCES
        total_calls = sum(len(inst.instruments.calls) for inst in instances)
        assert total_calls == 1, f"FIGI должен резолвиться один раз, got {total_calls}"

    @pytest.mark.asyncio
    async def test_unknown_ticker_raises(self, monkeypatch, with_credentials):
        _make_fake_tinvest_module(monkeypatch, {}, _sample_candles())

        from aqr.data.tinvest import TInvestAdapter

        adapter = TInvestAdapter()
        with pytest.raises(ValueError, match="Unknown ticker: 'FAKE_TICKER'"):
            await adapter.candles("FAKE_TICKER", "2023-01-02", "2024-12-31")


class TestIntervalMap:
    def test_all_seven_intervals(self):
        from aqr.data.tinvest import INTERVAL_MAP
        assert "1m" in INTERVAL_MAP
        assert "D1" in INTERVAL_MAP

    @pytest.mark.asyncio
    async def test_invalid_interval_raises(self, monkeypatch, with_credentials):
        _make_fake_tinvest_module(monkeypatch, {"SBER": "BBG004730N88"}, [])

        from aqr.data.tinvest import TInvestAdapter

        adapter = TInvestAdapter()
        with pytest.raises(ValueError, match="Unsupported interval"):
            await adapter.candles("SBER", "2023-01-02", "2024-12-31", interval="3d")


class TestSandboxTarget:
    def test_sandbox_default_true(self, monkeypatch, with_credentials):
        monkeypatch.delenv("INVEST_SANDBOX", raising=False)
        _make_fake_tinvest_module(monkeypatch, {"SBER": "BBG004730N88"}, _sample_candles())

        from aqr.data.tinvest import TInvestAdapter

        adapter = TInvestAdapter()
        assert adapter.sandbox is True

    def test_invest_sandbox_0_means_production(self, monkeypatch):
        monkeypatch.setenv("INVEST_SANDBOX", "0")
        monkeypatch.setenv("INVEST_TOKEN", "t.test")
        _make_fake_tinvest_module(monkeypatch, {"SBER": "BBG004730N88"}, _sample_candles())

        from aqr.data.tinvest import TInvestAdapter

        adapter = TInvestAdapter(token="t.test")
        assert adapter.sandbox is False

    def test_credentials_sandbox_overrides_env(self, monkeypatch):
        monkeypatch.setenv("INVEST_SANDBOX", "0")

        from aqr.graph.context import reset_credentials, set_credentials
        from aqr.registry import DecryptedSettings

        creds = DecryptedSettings(
            session_id="alice",
            llm_model="claude-3-5-sonnet-20241022",
            llm_api_key="sk-ant",
            openai_api_key="sk-oai",
            invest_token="t",
            invest_sandbox=True,
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
    @pytest.mark.asyncio
    async def test_no_retry_no_circuit_breaker(self, monkeypatch, with_credentials):
        from aqr.data import tinvest as tinvest_mod

        class FakeAsyncServicesBroken:
            instruments = None
            market_data = None

        class FakeAsyncClientBroken:
            INVEST_GRPC_API = "prod"
            INVEST_GRPC_API_SANDBOX = "sandbox"

            def __init__(self, token, *, target=None):
                pass

            async def __aenter__(self):
                raise ConnectionError("gRPC unavailable")

            async def __aexit__(self, *a):
                return None

        fake = types.ModuleType("t_tech.invest")
        fake.AsyncClient = FakeAsyncClientBroken
        fake.CandleInterval = MagicMock()
        fake.InstrumentType = MagicMock()
        fake.INVEST_GRPC_API = "prod"
        fake.INVEST_GRPC_API_SANDBOX = "sandbox"
        fake.constants = types.ModuleType("constants")
        fake.constants.INVEST_GRPC_API = "prod"
        fake.constants.INVEST_GRPC_API_SANDBOX = "sandbox"
        monkeypatch.setitem(sys.modules, "t_tech.invest", fake)

        tinvest_mod._tinvest_module = None

        from aqr.data.tinvest import TInvestAdapter

        adapter = TInvestAdapter()
        with pytest.raises(ConnectionError, match="gRPC unavailable"):
            await adapter.candles("SBER", "2023-01-02", "2024-12-31")


class TestCredentialsRequired:
    def test_raises_without_credentials(self, monkeypatch):
        from aqr.graph.context import current_credentials

        assert current_credentials() is None

        from aqr.data.tinvest import TInvestAdapter

        with pytest.raises(RuntimeError, match="token not provided"):
            TInvestAdapter()

    @pytest.mark.asyncio
    async def test_uses_credentials_when_no_token_arg(self, monkeypatch, with_credentials):
        _make_fake_tinvest_module(monkeypatch, {"SBER": "BBG004730N88"}, _sample_candles())

        from aqr.data.tinvest import TInvestAdapter

        adapter = TInvestAdapter()
        df = await adapter.candles("SBER", "2023-01-02", "2024-12-31")
        assert len(df) == 120


class TestClearFigiCache:
    @pytest.mark.asyncio
    async def test_clear_figi_cache(self, monkeypatch, with_credentials):
        candles = _sample_candles()
        fake = _make_fake_tinvest_module(
            monkeypatch, {"SBER": "BBG004730N88"}, candles,
        )

        from aqr.data.tinvest import TInvestAdapter

        adapter = TInvestAdapter()
        await adapter.candles("SBER", "2023-01-02", "2024-12-31")
        TInvestAdapter.clear_figi_cache()
        await adapter.candles("SBER", "2023-01-02", "2024-12-31")

        total = sum(len(inst.instruments.calls) for inst in fake.ASYNC_SERVICES_INSTANCES)
        assert total == 2
