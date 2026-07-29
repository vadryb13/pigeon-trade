"""Тесты для /chat/{token}/settings — форма, POST, status endpoint."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _stable_secret(monkeypatch):
    """Зафиксировать AQR_SESSION_SECRET ≥32 chars для воспроизводимости токенов."""
    monkeypatch.setenv("AQR_SESSION_SECRET", "test-secret-padded-to-32-bytes-base64==")


@pytest.fixture
def mock_db(monkeypatch):
    """Мок async_session_factory с in-memory store для SessionSettings."""

    class _FakeSession:
        def __init__(self):
            self._store: dict[tuple[type, str], object] = {}
            self.commits = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        def add(self, obj):
            if hasattr(obj, "session_id"):
                self._store[(type(obj), obj.session_id)] = obj

        async def get(self, model, key):
            return self._store.get((model, key))

        async def flush(self):
            return None

        async def commit(self):
            self.commits += 1

    db = _FakeSession()

    class _Factory:
        def __call__(self):
            return db

    monkeypatch.setattr("aqr.chat.web.async_session_factory", _Factory())
    return db


@pytest.fixture
def alice_token(monkeypatch):
    monkeypatch.setenv("AQR_SESSION_SECRET", "test-secret-padded-to-32-bytes-base64==")
    from aqr.auth import sign_session
    return sign_session("alice")


class TestSettingsPageGET:
    def test_unconfigured_returns_form(self, mock_db, alice_token):
        """Без session_settings — GET отдаёт HTML-форму."""
        from fastapi.testclient import TestClient

        from aqr.main import app

        client = TestClient(app)
        r = client.get(f"/chat/{alice_token}/settings")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        body = r.text
        assert "llm_model" in body
        assert "llm_api_key" in body
        assert "openai_api_key" in body
        assert "invest_token" in body
        assert "invest_sandbox" in body

    def test_already_configured_redirects_to_chat(self, mock_db, alice_token):
        """Если settings уже есть — редирект на /chat/{token}."""
        from datetime import UTC, datetime

        from aqr.crypto import encrypt_str
        from aqr.registry import SessionSettings

        mock_db._store[(SessionSettings, "alice")] = SessionSettings(
            session_id="alice",
            llm_model="claude-3-5-sonnet-20241022",
            llm_api_key_encrypted=encrypt_str("k1"),
            openai_api_key_encrypted=encrypt_str("k2"),
            invest_token_encrypted=encrypt_str("t1"),
            invest_sandbox=True,
            updated_at=datetime.now(UTC),
        )

        from fastapi.testclient import TestClient

        from aqr.main import app

        client = TestClient(app)
        r = client.get(f"/chat/{alice_token}/settings", follow_redirects=False)
        assert r.status_code == 303
        assert r.headers["location"] == "/chat"

    def test_invalid_token_returns_403(self, mock_db):
        from fastapi.testclient import TestClient

        from aqr.main import app

        client = TestClient(app)
        r = client.get("/chat/invalid_token/settings")
        assert r.status_code == 403


class TestSettingsPagePOST:
    def test_valid_form_saves_and_redirects(self, mock_db, alice_token):
        from fastapi.testclient import TestClient

        from aqr.main import app

        client = TestClient(app)
        r = client.post(
            f"/chat/{alice_token}/settings",
            data={
                "llm_model": "claude-3-5-sonnet-20241022",
                "llm_api_key": "sk-ant-fake",
                "openai_api_key": "sk-oai-fake",
                "invest_token": "t.INVEST_TOKEN_fake",
                "invest_sandbox": "on",
            },
            follow_redirects=False,
        )
        assert r.status_code == 303
        assert r.headers["location"] == "/chat"

        from aqr.registry import SessionSettings

        saved = mock_db._store.get((SessionSettings, "alice"))
        assert saved is not None
        assert saved.llm_model == "claude-3-5-sonnet-20241022"
        assert saved.invest_sandbox is True
        assert "sk-ant-fake" not in saved.llm_api_key_encrypted  # зашифровано

    def test_sandbox_unchecked(self, mock_db, alice_token):
        from fastapi.testclient import TestClient

        from aqr.main import app

        client = TestClient(app)
        r = client.post(
            f"/chat/{alice_token}/settings",
            data={
                "llm_model": "gpt-4o-mini",
                "llm_api_key": "sk-1",
                "openai_api_key": "sk-2",
                "invest_token": "t.INVEST_TOKEN",
                "invest_sandbox": "off",
            },
            follow_redirects=False,
        )
        assert r.status_code == 303

        from aqr.registry import SessionSettings

        saved = mock_db._store[(SessionSettings, "alice")]
        assert saved.invest_sandbox is False

    def test_missing_required_field_returns_422(self, mock_db, alice_token):
        from fastapi.testclient import TestClient

        from aqr.main import app

        client = TestClient(app)
        r = client.post(
            f"/chat/{alice_token}/settings",
            data={
                "llm_model": "claude-3-5-sonnet-20241022",
                # llm_api_key отсутствует
                "openai_api_key": "sk-2",
                "invest_token": "t.INVEST_TOKEN",
            },
        )
        assert r.status_code == 422

    def test_invalid_token_returns_403(self, mock_db):
        from fastapi.testclient import TestClient

        from aqr.main import app

        client = TestClient(app)
        r = client.post(
            "/chat/invalid_token/settings",
            data={
                "llm_model": "claude-3-5-sonnet-20241022",
                "llm_api_key": "sk-1",
                "openai_api_key": "sk-2",
                "invest_token": "t.INVEST_TOKEN",
            },
        )
        assert r.status_code == 403


class TestSettingsStatus:
    def test_unconfigured_returns_false(self, mock_db, alice_token):
        from fastapi.testclient import TestClient

        from aqr.main import app

        client = TestClient(app)
        r = client.get(f"/chat/{alice_token}/settings/status")
        assert r.status_code == 200
        assert r.json() == {"configured": False}

    def test_configured_returns_model(self, mock_db, alice_token):
        from datetime import UTC, datetime

        from aqr.crypto import encrypt_str
        from aqr.registry import SessionSettings

        mock_db._store[(SessionSettings, "alice")] = SessionSettings(
            session_id="alice",
            llm_model="gpt-4o-mini",
            llm_api_key_encrypted=encrypt_str("k1"),
            openai_api_key_encrypted=encrypt_str("k2"),
            invest_token_encrypted=encrypt_str("t1"),
            invest_sandbox=False,
            updated_at=datetime.now(UTC),
        )

        from fastapi.testclient import TestClient

        from aqr.main import app

        client = TestClient(app)
        r = client.get(f"/chat/{alice_token}/settings/status")
        assert r.status_code == 200
        data = r.json()
        assert data["configured"] is True
        assert data["llm_model"] == "gpt-4o-mini"
