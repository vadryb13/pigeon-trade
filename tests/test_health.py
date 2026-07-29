"""Тесты health-check эндпоинтов /health и /health/ready."""
from __future__ import annotations


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
    def test_ready_200_when_validate_passes(self, monkeypatch):
        """/health/ready → 200 если validate_runtime() возвращает ok."""
        from fastapi.testclient import TestClient

        async def fake_validate():
            return {"status": "ready", "postgres": "ok"}

        monkeypatch.setattr("aqr.main.validate_runtime", fake_validate)

        from aqr.main import app

        client = TestClient(app)
        r = client.get("/health/ready")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ready"
        assert data["postgres"] == "ok"

    def test_ready_503_when_validate_raises(self, monkeypatch):
        """/health/ready → 503 если validate_runtime() бросает RuntimeError."""
        from fastapi.testclient import TestClient

        async def fake_validate():
            raise RuntimeError("DATABASE_URL is required")

        monkeypatch.setattr("aqr.main.validate_runtime", fake_validate)

        from aqr.main import app

        client = TestClient(app)
        r = client.get("/health/ready")
        assert r.status_code == 503
        data = r.json()
        assert data["status"] == "degraded"
        assert "DATABASE_URL" in data["error"]
