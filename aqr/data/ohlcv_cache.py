"""DuckDB-кэш для OHLCV по тикерам.

Лениво импортирует duckdb — только при первом обращении (опциональная [data] extra).
Повторные прогоны на тех же тикерах/периодах не ходят в T-Invest gRPC.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


logger = logging.getLogger(__name__)

# DuckDB rejects concurrent connections to one file when their configuration
# differs (read-only versus read-write). The application cache is process-local,
# therefore a re-entrant process lock makes every schema/read/write operation
# atomic while preserving the synchronous cache API used via asyncio.to_thread.
_DUCKDB_LOCK = threading.RLock()


def _synchronized(method):
    def wrapped(*args, **kwargs):
        with _DUCKDB_LOCK:
            return method(*args, **kwargs)

    return wrapped


def _safe_float(value: Any) -> float | None:
    """Конвертация в float с защитой от NaN/None. Возвращает None для мусорных значений."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ohlcv (
    ticker      VARCHAR NOT NULL,
    timeframe   VARCHAR NOT NULL,
    begin       TIMESTAMP NOT NULL,
    open        DOUBLE,
    high        DOUBLE,
    low         DOUBLE,
    close       DOUBLE NOT NULL,
    volume      BIGINT,
    value       DOUBLE,
    fetched_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, timeframe, begin)
);
CREATE INDEX IF NOT EXISTS ix_ohlcv_ticker_tf ON ohlcv(ticker, timeframe, begin);
"""


class OhlcvCache:
    """Read-through кэш OHLCV с диском на DuckDB.

    Не создаёт файл, пока не вызван `put_cache` или `get_cached` (первый вызов
    открывает соединение и накатывает схему).

    Args:
        db_path: путь к DuckDB-файлу. По умолчанию — из env `AQR_CACHE_DIR`
            (дефолт `~/.aqr/ohlcv_cache.duckdb`), что обеспечивает одинаковый
            путь независимо от CWD. Родительский каталог создаётся автоматически.
    """

    DEFAULT_DIR = Path.home() / ".aqr"
    DEFAULT_BASENAME = "ohlcv_cache.duckdb"

    def __init__(
        self,
        db_path: str | Path | None = None,
    ) -> None:
        if db_path is None:
            env_dir = os.getenv("AQR_CACHE_DIR")
            base_dir = Path(env_dir) if env_dir else self.DEFAULT_DIR
            db_path = base_dir / self.DEFAULT_BASENAME
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False

    @_synchronized
    def _ensure_schema(self) -> None:
        """Инициализация схемы (один раз за инстанс)."""
        if self._initialized:
            return
        import duckdb
        with duckdb.connect(str(self.db_path)) as conn:
            conn.execute(SCHEMA_SQL)
        self._initialized = True

    @_synchronized
    def get_cached(
        self,
        ticker: str,
        start: str,
        end: str,
        timeframe: str = "D",
    ) -> pd.DataFrame | None:
        """Вернуть подмножество ряда в окне [start, end].

        Returns:
            DataFrame с колонками `close, open, high, low, volume, value` и
            DatetimeIndex `begin`, отсортированный по времени. `None`,
            если в кэше нет ни одной строки для данного тикера.
        """
        import pandas as pd

        self._ensure_schema()

        import duckdb
        with duckdb.connect(str(self.db_path), read_only=True) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM ohlcv WHERE ticker = ? AND timeframe = ? "
                "AND begin BETWEEN ? AND ?",
                [ticker, timeframe, start, end],
            ).fetchone()
            if row is None or row[0] == 0:
                return None

            df = conn.execute(
                "SELECT begin, open, high, low, close, volume, value "
                "FROM ohlcv WHERE ticker = ? AND timeframe = ? "
                "AND begin BETWEEN ? AND ? "
                "ORDER BY begin ASC",
                [ticker, timeframe, start, end],
            ).fetch_df()

        df["begin"] = pd.to_datetime(df["begin"])
        df = df.set_index("begin").sort_index()
        return df

    @_synchronized
    def put_cache(
        self,
        ticker: str,
        df: pd.DataFrame,
        timeframe: str = "D",
    ) -> int:
        """Upsert: добавить или обновить строки для тикера.

        Ожидаемые колонки в `df`: `open, high, low, close, volume, value`
        и индекс `begin` (DatetimeIndex). Если колонок нет — добавляются NaN.

        Returns:
            количество строк, реально вставленных/обновлённых.
        """
        import pandas as pd

        self._ensure_schema()

        if df.empty:
            return 0

        # Нормализация индекса и колонок
        if not isinstance(df.index, pd.DatetimeIndex):
            if "begin" in df.columns:
                df = df.set_index(pd.to_datetime(df["begin"]))
            elif "time" in df.columns:
                df = df.set_index(pd.to_datetime(df["time"]))
            else:
                raise ValueError(
                    "DataFrame должен иметь DatetimeIndex или колонку begin/time"
                )

        rows = []
        now = datetime.now(UTC).isoformat()
        skipped = 0
        for ts, row in df.iterrows():
            close_val = row.get("close")
            # NaN/None close → NOT NULL constraint violation → пропускаем бар
            if close_val is None or pd.isna(close_val):
                skipped += 1
                continue
            close_f = float(close_val)
            if pd.isna(close_f):
                skipped += 1
                continue

            rows.append((
                ticker, timeframe, pd.Timestamp(ts).to_pydatetime(),
                _safe_float(row.get("open")),
                _safe_float(row.get("high")),
                _safe_float(row.get("low")),
                close_f,
                int(row["volume"]) if "volume" in row
                                     and row["volume"] is not None
                                     and not pd.isna(row["volume"]) else None,
                _safe_float(row.get("value")),
                now,
            ))
        if skipped:
            logger.warning(
                "ohlcv_cache_skipped_nan_rows",
                extra={"ticker": ticker, "timeframe": timeframe, "skipped": skipped},
            )

        import duckdb
        with duckdb.connect(str(self.db_path)) as conn:
            conn.executemany(
                """
                INSERT INTO ohlcv
                    (ticker, timeframe, begin, open, high, low, close, volume, value, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (ticker, timeframe, begin) DO UPDATE SET
                    open = excluded.open,
                    high = excluded.high,
                    low = excluded.low,
                    close = excluded.close,
                    volume = excluded.volume,
                    value = excluded.value,
                    fetched_at = excluded.fetched_at
                """,
                rows,
            )
        return len(rows)

    @_synchronized
    def invalidate(self, ticker: str | None = None) -> int:
        """Очистить кэш: по тикеру (если задан) или целиком.

        Returns:
            количество удалённых строк.
        """
        self._ensure_schema()

        import duckdb
        with duckdb.connect(str(self.db_path)) as conn:
            if ticker is None:
                row = conn.execute("SELECT COUNT(*) FROM ohlcv").fetchone()
                count = row[0] if row else 0
                conn.execute("DELETE FROM ohlcv")
            else:
                row = conn.execute(
                    "SELECT COUNT(*) FROM ohlcv WHERE ticker = ?", [ticker]
                ).fetchone()
                count = row[0] if row else 0
                conn.execute("DELETE FROM ohlcv WHERE ticker = ?", [ticker])
        return count

    @_synchronized
    def stats(self) -> dict[str, int]:
        """Сводка по кэшу: количество тикеров и строк."""
        self._ensure_schema()

        import duckdb
        with duckdb.connect(str(self.db_path), read_only=True) as conn:
            row = conn.execute(
                "SELECT COUNT(DISTINCT ticker), COUNT(*) FROM ohlcv"
            ).fetchone()
            tickers = int(row[0]) if row else 0
            rows = int(row[1]) if row else 0
        return {"tickers": tickers, "rows": rows}
