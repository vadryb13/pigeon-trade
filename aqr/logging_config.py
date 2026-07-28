"""Структурированное логирование для AQR.

По умолчанию — человекочитаемый формат (для dev/CLI).
Если задан `AQR_LOG_JSON=1` (или `setup_logging(json=True)`) — JSON-формат
по одной строке на событие, парсится ELK/Loki/Datadog из коробки.

Контракт полей (фиксирован в JsonFormatter):
    ts, level, logger, message,
    run_id, tool, duration_ms, status, error

Дополнительные `extra={...}` в `logger.info(...)` мерджатся как flat-поля
(ключи на верхнем уровне JSON).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime
from typing import Any

# Фиксированный набор полей для structured-логов
_RESERVED_LOGRECORD_ATTRS = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "asctime", "taskName",
}

# B23: ключи, которые НЕЛЬЗЯ логировать as-is. Их значения заменяются на
# "***" в JsonFormatter, чтобы API-ключи/токены не утекали в ELK/Loki.
_SENSITIVE_KEYS = frozenset({
    "api_key", "apikey", "api-key",
    "token", "access_token", "refresh_token", "bearer_token",
    "secret", "password", "credential", "credentials",
    "invest_token", "llm_api_key", "openai_api_key",
    "anthropic_api_key", "gigachat_credentials",
})


def _redact_value(key: str, value: Any) -> Any:
    """Заменить значение чувствительного поля на '***'."""
    if isinstance(value, str) and value:
        return "***"
    if isinstance(value, (list, tuple)):
        return "[redacted]"
    if isinstance(value, dict):
        return {"[redacted]": len(value)}
    return "***"


class JsonFormatter(logging.Formatter):
    """JSON-formatter для structured-логов.

    Каждый log record → одна JSON-строка с фиксированным набором полей.
    Любые `extra={...}`-аргументы из logger.info(...) добавляются как
    flat-поля на верхний уровень.

    B23: значения чувствительных ключей (api_key, token, и т.п.)
    заменяются на '***' перед записью.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Доп.поля из extra=
        for key, value in record.__dict__.items():
            if key in _RESERVED_LOGRECORD_ATTRS or key.startswith("_"):
                continue
            if key.lower() in _SENSITIVE_KEYS:
                payload[key] = _redact_value(key, value)
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = repr(value)

        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(
    level: str | int = "INFO",
    json: bool | None = None,
    stream=None,
) -> None:
    """Настроить root-логгер.

    Args:
        level: уровень логирования (строка или int)
        json: True → JSON, False → human-readable, None → авто по env AQR_LOG_JSON
        stream: куда писать (по умолчанию stderr)
    """
    if json is None:
        json = os.getenv("AQR_LOG_JSON", "").lower() in ("1", "true", "yes")

    root = logging.getLogger()
    root.setLevel(level)

    # Удаляем старые handlers (важно для повторных вызовов в тестах)
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(stream or sys.stderr)
    if json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
    root.addHandler(handler)


def log_tool_call(
    logger: logging.Logger,
    run_id: str,
    tool: str,
    duration_ms: float,
    status: str = "ok",
    error: str | None = None,
    **extra,
) -> None:
    """Удобный хелпер для единообразного structured-логирования вызовов тулов.

    Пример:
        log_tool_call(logger, run_id="r-1", tool="backtest_one",
                      duration_ms=123.4, status="ok")
    """
    level = logging.WARNING if status == "error" else logging.INFO
    logger.log(level, "tool_call", extra={
        "run_id": run_id,
        "tool": tool,
        "duration_ms": round(duration_ms, 2),
        "status": status,
        "error": error,
        **extra,
    })
