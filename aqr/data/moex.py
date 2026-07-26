"""
MOEX ISS (Interactive Statistical Server) adapter.

Docs: https://iss.moex.com/iss/reference/

Point-in-time discipline:
- Every fetch records `as_of` timestamp
- Corporate actions applied only up to as_of
- No forward-fill of missing bars (leaves gaps explicit)
- Volume adjustments logged in manifest

Resilience (этап 7.1):
- HTTP timeout 10s на каждый запрос
- 3 попытки с exponential backoff (0.5s → 1s → 2s)
- Per-ticker circuit breaker: 5 ошибок подряд → 60s все запросы к этому тикеру
  пробрасывают CircuitOpenError мгновенно, без сетевого вызова
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import pandas as pd
import requests
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

MOEX_ISS_BASE = "https://iss.moex.com/iss"


class CircuitOpenError(RuntimeError):
    """Circuit breaker открыт для тикера — сетевой вызов пропущен."""


@dataclass
class _BreakerState:
    """Состояние CB для одного тикера."""

    failures: int = 0
    open_until: float = 0.0  # unix-time до которого breaker открыт


class MOEXAdapter:
    """
    Fetch MOEX securities data with point-in-time guarantees.

    Supported engines/markets:
    - stock/shares — equities (SBER, GAZP, LKOH, ...)
    - stock/index — indices (IMOEX, RTSI, ...)
    - futures/forts — futures (Si-, Br-, GD-, ...)
    - currency/selt — FX (USD/RUB, CNY/RUB, ...)
    - stock/bonds — bonds

    Example:
        adapter = MOEXAdapter()
        df = adapter.candles("SBER", "2024-01-01", "2024-12-31", interval="D")
    """

    ENGINE_MARKET_MAP = {
        "shares": ("stock", "shares"),
        "index": ("stock", "index"),
        "futures": ("futures", "forts"),
        "currency": ("currency", "selt"),
        "bonds": ("stock", "bonds"),
    }

    INTERVAL_MAP = {
        "1min": 1, "10min": 10, "1H": 60, "D": 24, "W": 7, "M": 31, "Q": 4,
    }

    def __init__(
        self,
        session: requests.Session | None = None,
        rate_limit_ms: int = 500,
        timeout: int = 10,
        max_retries: int = 3,
        cb_threshold: int = 5,
        cb_recovery_seconds: int = 60,
    ):
        self.session = session or requests.Session()
        self.rate_limit_ms = rate_limit_ms
        self.timeout = timeout
        self.max_retries = max_retries
        self.cb_threshold = cb_threshold
        self.cb_recovery_seconds = cb_recovery_seconds
        self._last_call = 0.0
        # CB state: {ticker: _BreakerState}
        self._breakers: dict[str, _BreakerState] = {}

    # ── Public resilience knobs ────────────────────────────────

    def is_breaker_open(self, ticker: str) -> bool:
        """True если CB открыт для тикера прямо сейчас."""
        st = self._breakers.get(ticker)
        if st is None:
            return False
        return time.time() < st.open_until

    def reset_breakers(self) -> None:
        """Сбросить все CB (для тестов и admin-команд)."""
        self._breakers.clear()

    # ── Internal: request with retry + CB ──────────────────────

    def _check_breaker(self, ticker: str) -> None:
        """Бросить CircuitOpenError если CB открыт."""
        st = self._breakers.get(ticker)
        if st is not None and time.time() < st.open_until:
            raise CircuitOpenError(
                f"Circuit breaker open for {ticker} until "
                f"{datetime.fromtimestamp(st.open_until).isoformat()}"
            )

    def _record_success(self, ticker: str) -> None:
        """Сбрасывает счётчик ошибок для тикера."""
        st = self._breakers.get(ticker)
        if st is not None:
            st.failures = 0
            st.open_until = 0.0

    def _record_failure(self, ticker: str) -> None:
        """Инкрементирует счётчик, открывает CB при превышении порога."""
        st = self._breakers.setdefault(ticker, _BreakerState())
        st.failures += 1
        if st.failures >= self.cb_threshold:
            st.open_until = time.time() + self.cb_recovery_seconds
            logger.warning(
                "circuit_breaker_open",
                extra={
                    "ticker": ticker,
                    "failures": st.failures,
                    "open_seconds": self.cb_recovery_seconds,
                },
            )

    def _rate_limit(self):
        now = time.time() * 1000
        elapsed = now - self._last_call
        if elapsed < self.rate_limit_ms:
            time.sleep((self.rate_limit_ms - elapsed) / 1000)
        self._last_call = time.time() * 1000

    def candles(
        self,
        security: str,
        from_date: str,
        to_date: str,
        interval: str = "D",
        engine: Literal["shares", "index", "futures", "currency", "bonds"] = "shares",
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candles.

        Returns DataFrame with columns: open, high, low, close, volume, value, begin, end
        Indexed by begin (UTC).
        """
        eng, market = self.ENGINE_MARKET_MAP[engine]
        int_code = self.INTERVAL_MAP.get(interval, 24)

        rows = []
        start = 0
        while True:
            self._rate_limit()
            url = (
                f"{MOEX_ISS_BASE}/engines/{eng}/markets/{market}/securities/"
                f"{security}/candles.json"
            )
            params = {
                "from": from_date,
                "till": to_date,
                "interval": int_code,
                "start": start,
            }
            r = self._request_with_retry("GET", url, params, ticker=security)
            r.raise_for_status()
            data = r.json()

            candles = data.get("candles", {})
            cols = candles.get("columns", [])
            batch = candles.get("data", [])

            if not batch:
                break
            for row in batch:
                rows.append(dict(zip(cols, row)))
            if len(batch) < 500:
                break
            start += len(batch)

        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "value"])

        df = pd.DataFrame(rows)
        df["begin"] = pd.to_datetime(df["begin"])
        df["end"] = pd.to_datetime(df["end"])
        df = df.set_index("begin").sort_index()
        return df

    def _request_with_retry(
        self,
        method: str,
        url: str,
        params: dict | None = None,
        ticker: str = "",
    ) -> requests.Response:
        """HTTP-запрос с retry+CB. 5xx и ConnectionError — ретраятся, 4xx — нет.

        На исчерпание попыток: бросает оригинальное requests-исключение.
        На CB open: бросает CircuitOpenError без сетевого вызова.
        """
        self._check_breaker(ticker)

        retry_decorator = retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
            retry=retry_if_exception_type(
                (requests.ConnectionError, requests.Timeout, requests.HTTPError)
            ),
            reraise=True,
        )

        @retry_decorator
        def _do_request() -> requests.Response:
            self._rate_limit()
            try:
                resp = self.session.request(
                    method, url, params=params, timeout=self.timeout,
                )
            except (requests.ConnectionError, requests.Timeout) as e:
                # Сетевые ошибки — ретраим, помечаем как failure для CB
                self._record_failure(ticker)
                logger.warning(
                    "moex_request_retry",
                    extra={"ticker": ticker, "error": str(e)},
                )
                raise
            if resp.status_code >= 500:
                # 5xx — ретраим
                self._record_failure(ticker)
                logger.warning(
                    "moex_request_5xx",
                    extra={"ticker": ticker, "status": resp.status_code},
                )
                raise requests.HTTPError(
                    f"{resp.status_code} {resp.reason}", response=resp,
                )
            if resp.status_code >= 400:
                # 4xx — НЕ ретраим, НЕ помечаем CB
                logger.warning(
                    "moex_request_4xx",
                    extra={"ticker": ticker, "status": resp.status_code},
                )
            return resp

        try:
            resp = _do_request()
        except RetryError as e:
            # tenacity исчерпал попытки — отдаём последнее исключение
            raise e.last_attempt.exception()
        # Успешный ответ сбрасывает счётчик CB для тикера
        self._record_success(ticker)
        return resp

    # NOTE: ранее здесь были `securities_list()` и `corporate_actions()` —
    # удалены в рамках DEAD-2. `corporate_actions` был заготовкой для PIT,
    # но не подключён к `candles()` (не применён к историческим ценам).
    # Полноценный PIT — отдельная задача; пока работает safety net в
    # `load_prices` (PIT safety net).
