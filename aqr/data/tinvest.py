"""T-Invest gRPC-адаптер.

Основан на `t-tech-investments` (https://opensource.tbank.ru/invest/invest-python).
Lazy import `t_tech.invest` чтобы aqr.main стартовал без поднятия gRPC
зависимостей (см. AGENTS.md инвариант 7).

Строгий режим: одна попытка на запрос, без retry/circuit-breaker.
На любую ошибку (сеть, таймаут, неизвестный FIGI) — raise.

Per-session credentials (token + sandbox) берутся из ContextVar.

Concurrency: `_resolve_figi` защищён `threading.Lock` — load_prices
гоняет несколько `adapter.candles(...)` через `asyncio.to_thread` → они
выполняются параллельно в ThreadPoolExecutor и без лока race-ят на
class-level `_figi_cache` (B5). В CPython dict.get/set атомарны на уровне
GIL, но между get-проверкой и set может вклиниться другой поток, и
тогда оба сходят в gRPC InstrumentsService.
"""
from __future__ import annotations

import os
import sys
import threading
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from t_tech.invest import CandleInterval, Client


# Lazy import для тестов и для отделения import-time от runtime
_tinvest_module = None


def _get_tinvest():
    """Lazy import t_tech.invest. На ошибке ImportError — RuntimeError.

    Также проверяем sys.modules — позволяет тестам подменять
    `t_tech.invest` через `monkeypatch.setitem(sys.modules, ...)`.
    """
    global _tinvest_module
    if _tinvest_module is not None:
        return _tinvest_module

    # Тестовая подмена через sys.modules — избегаем реального import
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


class TInvestAdapter:
    """Обёртка над t_tech.invest.Client: candles + FIGI-resolution.

    INTERVAL_MAP — 7 валидных интервалов через T-Invest CandleInterval.
    Любой другой ключ → ValueError (см. AGENTS.md gotcha).
    """

    # Class-level cache для ticker → FIGI. Сбрасывается через clear_figi_cache.
    _figi_cache: dict[str, str] = {}
    # threading.Lock защищает _figi_cache от concurrent writers из
    # asyncio.to_thread-пула (B5). Class-level — все инстансы делят один лок.
    _figi_lock: threading.Lock = threading.Lock()

    def __init__(
        self,
        token: str | None = None,
        sandbox: bool | None = None,
    ) -> None:
        # Если token не передан — берём из per-session ContextVar
        if token is None:
            from aqr.agent.context import current_credentials

            creds = current_credentials()
            if creds is None:
                raise RuntimeError(
                    "TInvestAdapter: token not provided and no session "
                    "credentials in context. Configure via /chat/{token}/settings."
                )
            token = creds.invest_token
            sandbox = creds.invest_sandbox

        # Sandbox по дефолту для dev/CI
        if sandbox is None:
            sandbox = os.getenv("INVEST_SANDBOX", "1") != "0"

        ti = _get_tinvest()
        self.token = token
        self.sandbox = sandbox
        self._target = (
            ti.INVEST_GRPC_API_SANDBOX if sandbox else ti.INVEST_GRPC_API
        )
        self._Client = ti.Client
        self._CandleInterval = ti.CandleInterval

    def candles(
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

        figi = self._resolve_figi(ticker)
        interval_enum = getattr(
            self._CandleInterval, INTERVAL_MAP[interval]
        )

        with self._Client(self.token, target=self._target) as client:
            response = client.market_data.get_candles(
                figi=figi,
                from_=_parse_dt(from_date),
                to=_parse_dt(to_date),
                interval=interval_enum,
            )

        return _candles_to_dataframe(response.candles)

    def _resolve_figi(self, ticker: str) -> str:
        """Lazy lookup ticker → FIGI через InstrumentsService.

        Результат кэшируется в class-level dict. Неизвестный тикер → ValueError.
        Защищён `threading.Lock` от concurrent writers (B5): asyncio.to_thread
        разносит sync-вызовы по ThreadPoolExecutor, и без лока оба потока
        могут одновременно сходить в InstrumentsService.
        """
        # Fast-path: cache hit без лока (dict.get — атомарный в CPython).
        cached = TInvestAdapter._figi_cache.get(ticker)
        if cached is not None:
            return cached

        with TInvestAdapter._figi_lock:
            # Double-check под локом — другой поток мог уже заполнить.
            cached = TInvestAdapter._figi_cache.get(ticker)
            if cached is not None:
                return cached
            with self._Client(self.token, target=self._target) as client:
                r = client.instruments.get_instrument_by_ticker(
                    ticker=ticker,
                    class_code="TQBR",
                )
            instruments = list(r.instruments)
            if not instruments:
                raise ValueError(f"Unknown ticker: {ticker!r}")
            TInvestAdapter._figi_cache[ticker] = instruments[0].figi
            return TInvestAdapter._figi_cache[ticker]

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
    # Quotation.units — целая часть, Quotation.nano — дробная (10^-9)
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
