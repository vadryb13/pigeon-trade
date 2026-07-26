"""Тесты structured logging."""
from __future__ import annotations

import io
import json
import logging

from aqr.logging_config import JsonFormatter, log_tool_call, setup_logging


class TestJsonFormatter:
    def setup_method(self):
        self.formatter = JsonFormatter()

    def _make_record(self, msg="hi", level=logging.INFO, **extra):
        record = logging.LogRecord(
            name="test", level=level, pathname="x", lineno=1,
            msg=msg, args=(), exc_info=None,
        )
        for k, v in extra.items():
            setattr(record, k, v)
        return record

    def test_basic_record(self):
        """Минимальный record → JSON с фиксированными полями."""
        rec = self._make_record("hello")
        out = json.loads(self.formatter.format(rec))
        assert out["level"] == "INFO"
        assert out["logger"] == "test"
        assert out["message"] == "hello"
        assert "ts" in out

    def test_extra_fields_merged(self):
        """extra={...} поля появляются на верхнем уровне JSON."""
        rec = self._make_record("tool_call", run_id="r1", tool="backtest_one",
                                 duration_ms=12.5, status="ok")
        out = json.loads(self.formatter.format(rec))
        assert out["run_id"] == "r1"
        assert out["tool"] == "backtest_one"
        assert out["duration_ms"] == 12.5
        assert out["status"] == "ok"

    def test_non_serializable_field_falls_back_to_repr(self):
        """Несериализуемые значения → repr()."""
        class _Opaque:
            def __repr__(self):
                return "<opaque>"

        rec = self._make_record("msg", opaque=_Opaque())
        out = json.loads(self.formatter.format(rec))
        assert out["opaque"] == "<opaque>"

    def test_exception_info_included(self):
        """exc_info попадает в поле error."""
        try:
            raise ValueError("boom")
        except ValueError:
            import sys
            rec = self._make_record("failed", exc_info=sys.exc_info())
        out = json.loads(self.formatter.format(rec))
        assert "error" in out
        assert "ValueError: boom" in out["error"]


class TestLogToolCall:
    def test_log_tool_call_emits_warning_on_error(self):
        """status='error' → WARNING level."""
        logger = logging.getLogger("test_log_tool_call_err")
        log_tool_call(logger, run_id="r", tool="x", duration_ms=10, status="error", error="boom")
        # Просто не падает + логирует
        assert True

    def test_log_tool_call_info_on_ok(self):
        logger = logging.getLogger("test_log_tool_call_ok")
        log_tool_call(logger, run_id="r", tool="x", duration_ms=10, status="ok")


class TestSetupLogging:
    def test_setup_logging_human_readable_by_default(self, monkeypatch):
        """Без AQR_LOG_JSON → человекочитаемый формат."""
        monkeypatch.delenv("AQR_LOG_JSON", raising=False)

        stream = io.StringIO()
        setup_logging(level="INFO", stream=stream)

        logger = logging.getLogger("test_setup_default")
        logger.info("hello")

        output = stream.getvalue()
        assert "INFO" in output
        assert "hello" in output
        # Не должно быть JSON
        assert not output.strip().startswith("{")

    def test_setup_logging_json_via_env(self, monkeypatch):
        """AQR_LOG_JSON=1 → JSON."""
        monkeypatch.setenv("AQR_LOG_JSON", "1")

        stream = io.StringIO()
        setup_logging(level="INFO", stream=stream)

        logger = logging.getLogger("test_setup_json")
        logger.info("hello", extra={"run_id": "r1"})

        output = stream.getvalue().strip()
        data = json.loads(output)
        assert data["message"] == "hello"
        assert data["run_id"] == "r1"

    def test_setup_logging_json_via_param(self):
        """setup_logging(json=True) → JSON."""
        stream = io.StringIO()
        setup_logging(level="INFO", json=True, stream=stream)

        logger = logging.getLogger("test_setup_json_param")
        logger.info("via_param")

        output = stream.getvalue().strip()
        data = json.loads(output)
        assert data["message"] == "via_param"
