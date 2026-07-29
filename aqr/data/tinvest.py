"""T-Invest gRPC-адаптер.

Основан на `t-tech-investments` (https://opensource.tbank.ru/invest/invest-python).
Lazy import `t_tech.invest` чтобы aqr.main стартовал без поднятия gRPC
зависимостей (см. AGENTS.md инвариант 7).

Строгий режим: одна попытка на запрос, без retry/circuit-breaker.
На любую ошибку (сеть, таймаут, неизвестный FIGI) — raise.

Per-session credentials (token + sandbox) берутся из ContextVar или env.

Новый SDK (1.0.0+): `AsyncClient` как async context manager,
`find_instrument` с фильтром по `instrument_kind` для поиска FIGI,
`get_candles` для получения свечей.

SSL: `SSL_TBANK_VERIFY=true` использует сертификат МинЦифры
из пакета `t_tech.invest.certs`.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from t_tech.invest import AsyncClient

# Lazy import для тестов и для отделения import-time от runtime
_tinvest_module = None


def _get_tinvest():
    """Lazy import t_tech.invest. На ошибке ImportError — RuntimeError.

    Проверяет `sys.modules` перед импортом — позволяет тестам подменять
    t_tech.invest через monkeypatch.setitem(sys.modules, "t_tech.invest", fake).
    """
    global _tinvest_module
    if _tinvest_module is not None:
        return _tinvest_module

    # Тестовая подмена через sys.modules — проверяем ДО реального import,
    # чтобы тестовый fake перекрывал реальный пакет.
    ti = sys.modules.get("t_tech.invest")
    if ti is not None:
        _tinvest_module = ti
        return _tinvest_module

    try:
        import t_tech.invest  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "t_tech.invest not installed. Install with: "
            "pip install t-tech-investments "
            "--index-url https://opensource.tbank.ru/api/v4/projects/238/packages/pypi/simple"
        ) from e
    _tinvest_module = t_tech.invest
    return _tinvest_module


# Маппинг строковых интервалов на t_tech.invest.CandleInterval
INTERVAL_MAP: dict[str, str] = {
    "1m": "CANDLE_INTERVAL_1_MIN",
    "5m": "CANDLE_INTERVAL_5_MIN",
    "15m": "CANDLE_INTERVAL_15_MIN",
    "H1": "CANDLE_INTERVAL_HOUR",
    "D1": "CANDLE_INTERVAL_DAY",
    "W": "CANDLE_INTERVAL_WEEK",
    "M": "CANDLE_INTERVAL_MONTH",
}
# Производные интервалы (новый SDK)
_EXTRA_INTERVAL_MAP: dict[str, str] = {
    "2m": "CANDLE_INTERVAL_2_MIN",
    "3m": "CANDLE_INTERVAL_3_MIN",
    "10m": "CANDLE_INTERVAL_10_MIN",
    "30m": "CANDLE_INTERVAL_30_MIN",
    "2H": "CANDLE_INTERVAL_2_HOUR",
    "4H": "CANDLE_INTERVAL_4_HOUR",
}
INTERVAL_MAP.update(_EXTRA_INTERVAL_MAP)


class TInvestAdapter:
    """Обёртка над t_tech.invest.AsyncClient: candles + FIGI-resolution.

    INTERVAL_MAP — все валидные интервалы через T-Invest CandleInterval.
    Любой другой ключ → KeyError (см. AGENTS.md gotcha).
    """

    # Class-level cache для ticker → FIGI. Сбрасывается через clear_figi_cache.
    _figi_cache: dict[str, str] = {}

    def __init__(
        self,
        token: str | None = None,
        sandbox: bool | None = None,
    ) -> None:
        # Если token не передан — берём из per-session ContextVar
        if token is None:
            from aqr.graph.context import current_credentials

            creds = current_credentials()
            if creds is not None:
                token = creds.invest_token
                sandbox = creds.invest_sandbox
            # Fallback: env
            if token is None:
                token = os.getenv("INVEST_TOKEN")
            if token is None:
                raise RuntimeError(
                    "TInvestAdapter: token not provided and no session "
                    "credentials in context. Configure via /chat/{token}/settings "
                    "or set INVEST_TOKEN env var."
                )

        # Sandbox по дефолту для dev/CI
        if sandbox is None:
            sandbox = os.getenv("INVEST_SANDBOX", "1") != "0"

        self.token = token
        self.sandbox = sandbox

        # Для market data используем production endpoint
        # (sandbox endpoint не отдаёт свечи, но песочные токены
        # работают и с production target для данных)
        ti = _get_tinvest()
        self._target = os.getenv("INVEST_GRPC_API") or ti.constants.INVEST_GRPC_API

    def _build_async_client(self) -> "AsyncClient":
        """Создать AsyncClient на основе переданного token и target."""
        from t_tech.invest import AsyncClient

        return AsyncClient(token=self.token, target=self._target)

    async def candles(
        self,
        ticker: str,
        from_date: str,
        to_date: str,
        interval: str = "D1",
    ) -> pd.DataFrame:
        """OHLCV свечи для тикера.

        Returns: DataFrame с колонками [open, high, low, close, volume]
        и индексом времени (UTC).
        """
        if interval not in INTERVAL_MAP:
            raise ValueError(
                f"Unsupported interval: {interval!r}. "
                f"Valid: {sorted(INTERVAL_MAP.keys())}"
            )

        from t_tech.invest import CandleInterval

        figi = await self._resolve_figi(ticker)
        interval_enum = getattr(CandleInterval, INTERVAL_MAP[interval])

        async with self._build_async_client() as client:
            response = await client.market_data.get_candles(
                figi=figi,
                from_=_parse_dt(from_date),
                to=_parse_dt(to_date),
                interval=interval_enum,
            )

        return _candles_to_dataframe(response.candles)

    async def _resolve_figi(self, ticker: str) -> str:
        """Lazy lookup ticker → FIGI через InstrumentsService.

        Результат кэшируется в class-level dict.
        Неизвестный тикер → ValueError.
        При нескольких FIGI (акция в разных режимах) выбираем
        Bloomberg FIGI (начинается с BBG) — основной для market data.
        """
        cached = TInvestAdapter._figi_cache.get(ticker)
        if cached is not None:
            return cached

        from t_tech.invest import InstrumentType

        async with self._build_async_client() as client:
            result = await client.instruments.find_instrument(
                query=ticker,
                instrument_kind=InstrumentType.INSTRUMENT_TYPE_SHARE,
            )
            instruments = [i for i in result.instruments if i.ticker == ticker]
            if not instruments:
                raise ValueError(f"Unknown ticker: {ticker!r}")
            # Предпочитаем Bloomberg FIGI (BBG...) — основной для market data
            figi = next((i.figi for i in instruments if i.figi.startswith("BBG")), instruments[0].figi)
            TInvestAdapter._figi_cache[ticker] = figi
            return figi

    @classmethod
    def clear_figi_cache(cls) -> None:
        """Очистить FIGI-кэш (для тестов)."""
        cls._figi_cache.clear()


def _parse_dt(s: str) -> datetime:
    """Парсит YYYY-MM-DD в timezone-aware datetime (UTC)."""
    if "T" in s:
        return datetime.fromisoformat(s)
    return datetime.fromisoformat(f"{s}T00:00:00+00:00")


def _quotation_to_float(q) -> float:
    """t_tech.invest возвращает цены как Quotation (units + nano).
    Конвертируем в float для pandas.
    """
    units = getattr(q, "units", 0)
    nano = getattr(q, "nano", 0)
    return float(units) + float(nano) / 1e9


def _candles_to_dataframe(candles) -> pd.DataFrame:
    """Конвертирует HistoricCandle-список в DataFrame [open, high, low, close, volume]."""
    rows = []
    for c in candles:
        rows.append({
            "time": c.time,
            "open": _quotation_to_float(c.open),
            "high": _quotation_to_float(c.high),
            "low": _quotation_to_float(c.low),
            "close": _quotation_to_float(c.close),
            "volume": c.volume,
        })

    if not rows:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"]
        ).set_index(pd.DatetimeIndex([], name="time"))

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    return df
