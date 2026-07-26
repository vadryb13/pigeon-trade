"""
Data manifest — записывает точно какие данные использовались для каждого backtest.

Позволяет reproducibility: код + seed + data_manifest_hash полностью
восстанавливают результат.

⚠️ EXPERIMENTAL: модуль не подключён к pipeline (см. DEAD-1 в REVIEW.md).
Импорт через `aqr.data.DataManifest` — lazy (`aqr/data/__init__.py`
через `__getattr__`). Не используйте в production-коде до того, как
будет спроектирован PIT-конвейер для дивидендов/сплитов.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import duckdb

MANIFEST_SCHEMA = """
CREATE TABLE IF NOT EXISTS data_snapshots (
    id            VARCHAR PRIMARY KEY,      -- hash содержимого
    source        VARCHAR NOT NULL,          -- 'moex', 'binance', etc.
    security      VARCHAR NOT NULL,
    interval      VARCHAR NOT NULL,
    from_date     DATE NOT NULL,
    to_date       DATE NOT NULL,
    as_of         TIMESTAMP NOT NULL,        -- когда был сделан snapshot
    n_rows        BIGINT,
    file_path     VARCHAR,                   -- путь к parquet
    schema_hash   VARCHAR,                   -- hash от column names/types
    content_hash  VARCHAR,                   -- hash от всех values
    metadata      JSON,                       -- corp_actions applied, currency, tz, etc.
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS backtest_data_lineage (
    backtest_id     VARCHAR NOT NULL,
    snapshot_id     VARCHAR NOT NULL,
    role            VARCHAR,                  -- 'primary', 'benchmark', 'universe'
    used_columns    VARCHAR[],
    PRIMARY KEY (backtest_id, snapshot_id)
);
"""


class DataManifest:
    """
    Единая точка учёта всех датасетов.

    Каждая гипотеза при backtest сохраняет lineage: какие snapshot_id использовала.
    """

    def __init__(self, db_path: str | Path = "workspace/data_manifest.duckdb"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self):
        with duckdb.connect(str(self.db_path)) as conn:
            conn.execute(MANIFEST_SCHEMA)

    def register_snapshot(
        self,
        source: str,
        security: str,
        interval: str,
        from_date: str,
        to_date: str,
        df,
        file_path: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """
        Register a data snapshot. Content-addressed by hash so identical fetches dedup.

        Returns:
            snapshot_id (16-char hex hash)
        """
        import pandas as pd

        # Content hash from DataFrame
        content_bytes = pd.util.hash_pandas_object(df, index=True).values.tobytes()
        content_hash = hashlib.sha256(content_bytes).hexdigest()[:32]

        schema_str = json.dumps({c: str(df[c].dtype) for c in df.columns}, sort_keys=True)
        schema_hash = hashlib.sha256(schema_str.encode()).hexdigest()[:16]

        snap_id = hashlib.sha256(
            f"{source}|{security}|{interval}|{from_date}|{to_date}|{content_hash}".encode()
        ).hexdigest()[:16]

        with duckdb.connect(str(self.db_path)) as conn:
            existing = conn.execute(
                "SELECT id FROM data_snapshots WHERE id = ?", [snap_id]
            ).fetchone()
            if existing:
                return snap_id

            conn.execute(
                """
                INSERT INTO data_snapshots
                (id, source, security, interval, from_date, to_date, as_of,
                 n_rows, file_path, schema_hash, content_hash, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    snap_id, source, security, interval, from_date, to_date,
                    datetime.utcnow(), len(df), file_path or "",
                    schema_hash, content_hash,
                    json.dumps(metadata or {}),
                ],
            )
        return snap_id

    def record_usage(
        self,
        backtest_id: str,
        snapshot_id: str,
        role: str = "primary",
        used_columns: list[str] | None = None,
    ):
        with duckdb.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO backtest_data_lineage
                (backtest_id, snapshot_id, role, used_columns) VALUES (?, ?, ?, ?)
                """,
                [backtest_id, snapshot_id, role, used_columns or []],
            )

    def reproduce_check(self, backtest_id: str) -> dict:
        """Return everything needed to reproduce a backtest."""
        with duckdb.connect(str(self.db_path)) as conn:
            rows = conn.execute(
                """
                SELECT s.id, s.source, s.security, s.interval,
                       s.from_date, s.to_date, s.content_hash, s.metadata,
                       l.role, l.used_columns
                FROM backtest_data_lineage l
                JOIN data_snapshots s ON s.id = l.snapshot_id
                WHERE l.backtest_id = ?
                """,
                [backtest_id],
            ).fetchall()
        return {
            "backtest_id": backtest_id,
            "snapshots": [
                {
                    "id": r[0], "source": r[1], "security": r[2], "interval": r[3],
                    "from_date": str(r[4]), "to_date": str(r[5]),
                    "content_hash": r[6], "metadata": json.loads(r[7]),
                    "role": r[8], "used_columns": r[9],
                }
                for r in rows
            ],
        }
