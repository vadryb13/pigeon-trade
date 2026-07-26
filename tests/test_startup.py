"""Тесты для aqr.startup.validate_runtime().

Проверяют:
- missing DATABASE_URL → RuntimeError
- missing/short AQR_SESSION_SECRET → RuntimeError
- успешный путь с моком docker/compose/pg
- timeout при недоступности Postgres
"""
from __future__ import annotations

import asyncio
import subprocess

import pytest


@pytest.fixture
def required_env(monkeypatch):
    """Минимальный набор env для прохождения _check_env."""
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@localhost/db")
    monkeypatch.setenv("AQR_SESSION_SECRET", "x" * 32)


class TestEnvChecks:
    async def test_missing_database_url_raises(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.setenv("AQR_SESSION_SECRET", "x" * 32)
        from aqr.startup import validate_runtime

        with pytest.raises(RuntimeError, match="DATABASE_URL is required"):
            await validate_runtime()

    async def test_missing_session_secret_raises(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@localhost/db")
        monkeypatch.delenv("AQR_SESSION_SECRET", raising=False)
        from aqr.startup import validate_runtime

        with pytest.raises(RuntimeError, match="AQR_SESSION_SECRET is required"):
            await validate_runtime()

    async def test_short_session_secret_raises(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@localhost/db")
        monkeypatch.setenv("AQR_SESSION_SECRET", "short")
        from aqr.startup import validate_runtime

        with pytest.raises(RuntimeError, match="≥32 chars"):
            await validate_runtime()

    async def test_both_missing_reports_both(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("AQR_SESSION_SECRET", raising=False)
        from aqr.startup import validate_runtime

        with pytest.raises(RuntimeError) as ei:
            await validate_runtime()
        msg = str(ei.value)
        assert "DATABASE_URL" in msg
        assert "AQR_SESSION_SECRET" in msg


class TestPostgresValidation:
    async def test_docker_unavailable_skips_compose(
        self, monkeypatch, required_env
    ):
        """Если docker недоступен — пропускаем compose, идём сразу в SELECT 1."""
        monkeypatch.setattr(
            "aqr.startup._docker_available", lambda: False
        )
        calls = {"compose": 0, "select": 0}

        def fake_compose(*a, **kw):
            calls["compose"] += 1
            return subprocess.CompletedProcess(a, 0, b"", b"")

        async def fake_wait(db_url):
            calls["select"] += 1
            return True, ""

        monkeypatch.setattr("aqr.startup._compose_up_postgres", fake_compose)
        monkeypatch.setattr("aqr.startup._wait_for_pg", fake_wait)

        from aqr.startup import validate_runtime

        result = await validate_runtime()
        assert result["status"] == "ready"
        assert calls["compose"] == 0  # compose не звался
        assert calls["select"] == 1

    async def test_compose_failure_raises(self, monkeypatch, required_env):
        monkeypatch.setattr("aqr.startup._docker_available", lambda: True)
        monkeypatch.setattr(
            "aqr.startup._compose_up_postgres",
            lambda: (False, "compose broken"),
        )
        from aqr.startup import validate_runtime

        with pytest.raises(RuntimeError, match="compose broken"):
            await validate_runtime()

    async def test_select1_timeout_raises(self, monkeypatch, required_env):
        monkeypatch.setattr("aqr.startup._docker_available", lambda: True)
        monkeypatch.setattr(
            "aqr.startup._compose_up_postgres",
            lambda: (True, ""),
        )
        monkeypatch.setattr(
            "aqr.startup._wait_for_pg",
            lambda url: asyncio.sleep(0, result=(False, "not ready")),
        )
        from aqr.startup import validate_runtime

        with pytest.raises(RuntimeError, match="not ready"):
            await validate_runtime()


class TestDockerAvailable:
    def test_docker_info_success(self, monkeypatch):
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **kw: subprocess.CompletedProcess(
                a[0], 0, b"Server: Docker", b""
            ),
        )
        from aqr.startup import _docker_available

        assert _docker_available() is True

    def test_docker_info_failure(self, monkeypatch):
        def fail(*a, **kw):
            raise subprocess.CalledProcessError(1, a[0])

        monkeypatch.setattr("subprocess.run", fail)
        from aqr.startup import _docker_available

        assert _docker_available() is False

    def test_docker_not_installed(self, monkeypatch):
        def fail(*a, **kw):
            raise FileNotFoundError("docker not found")

        monkeypatch.setattr("subprocess.run", fail)
        from aqr.startup import _docker_available

        assert _docker_available() is False

    def test_docker_timeout(self, monkeypatch):
        def fail(*a, **kw):
            raise subprocess.TimeoutExpired(a[0], 5)

        monkeypatch.setattr("subprocess.run", fail)
        from aqr.startup import _docker_available

        assert _docker_available() is False
