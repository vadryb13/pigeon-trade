"""Тесты health-check эндпоинтов /health и /health/ready."""
from __future__ import annotations

import pytest


@pytest.fixture
def mock_deps_ok(monkeypatch):
    """Мок БД и MOEX — оба отвечают OK."""
    from aqr import db as db_mod

    class _FakeResult:
        def scalar(self):
            return 1

    class _FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def execute(self, *args, **kw):
            return _FakeResult()

    class _FakeFactory:
        def __call__(self):
            return _FakeSession()

    monkeypatch.setattr(db_mod, "_async_session_factory", _FakeFactory())

    # Подменяем requests.head на «успех»
    monkeypatch.setattr(
        "aqr.main.requests.head",
        lambda *a, **kw: type("R", (), {"status_code": 200})(),
    )
    return _FakeFactory


class TestHealthLive:
    def test_health_always_ok(self):
        """/health — всегда 200, не зависит от внешних сервисов."""
        from fastapi.testclient import TestClient

        from aqr.main import app

        client = TestClient(app)
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        assert "version" in r.json()


class TestHealthReady:
    def test_ready_200_when_all_deps_up(self, mock_deps_ok):
        """/health/ready → 200 если Postgres и MOEX OK."""
        from fastapi.testclient import TestClient

        from aqr.main import app

        client = TestClient(app)
        r = client.get("/health/ready")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ready"
        assert data["postgres"] == "ok"
        assert data["moex"] == "ok"

    def test_ready_503_when_postgres_down(self, monkeypatch):
        """/health/ready → 503 если Postgres недоступен."""
        from aqr import db as db_mod

        class _BrokenFactory:
            def __call__(self):
                raise RuntimeError("DB down")

        monkeypatch.setattr(db_mod, "_async_session_factory", _BrokenFactory())
        # MOEX OK
        monkeypatch.setattr(
            "aqr.main.requests.head",
            lambda *a, **kw: type("R", (), {"status_code": 200})(),
        )

        from fastapi.testclient import TestClient

        from aqr.main import app

        client = TestClient(app)
        r = client.get("/health/ready")
        assert r.status_code == 503
        data = r.json()
        assert data["status"] == "degraded"
        assert "down" in data["postgres"].lower()
        assert data["moex"] == "ok"

    def test_ready_503_when_moex_down(self, mock_deps_ok):
        """/health/ready → 503 если MOEX HEAD возвращает не-200."""
        import requests as real_requests

        def fail_head(*a, **kw):
            raise real_requests.ConnectionError("moex down")

        import aqr.main
        aqr.main.requests.head = fail_head

        from fastapi.testclient import TestClient

        from aqr.main import app

        client = TestClient(app)
        r = client.get("/health/ready")
        assert r.status_code == 503
        data = r.json()
        assert data["status"] == "degraded"
        assert data["postgres"] == "ok"
        assert "down" in data["moex"].lower()

    def test_ready_503_when_moex_returns_500(self, monkeypatch):
        """/health/ready → 503 если MOEX HEAD вернул 500."""
        from aqr import db as db_mod

        class _FakeSession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            async def execute(self, *args, **kw):
                class _R:
                    def scalar(self):
                        return 1
                return _R()

        monkeypatch.setattr(
            db_mod, "_async_session_factory",
            type("_F", (), {"__call__": lambda self: _FakeSession()})(),
        )
        monkeypatch.setattr(
            "aqr.main.requests.head",
            lambda *a, **kw: type("R", (), {"status_code": 500})(),
        )

        from fastapi.testclient import TestClient

        from aqr.main import app

        client = TestClient(app)
        r = client.get("/health/ready")
        assert r.status_code == 503
        assert "down: HTTP 500" in r.json()["moex"]
