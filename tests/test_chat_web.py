"""Тесты для aqr/chat/web.py — Web UI endpoints."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _stable_secret(monkeypatch):
    """Зафиксировать AQR_SESSION_SECRET для воспроизводимости токенов."""
    monkeypatch.setenv("AQR_SESSION_SECRET", "test-secret-padded-to-32-bytes-base64==")


@pytest.fixture(autouse=True)
def _mock_db(monkeypatch):
    """Мок БД для WS integration-теста ниже."""
    from aqr import session as db_mod
    from aqr.chat import ws as ws_mod

    class _S:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return None
        async def commit(self):
            pass
        async def flush(self):
            pass
        async def get(self, *args, **kw):
            return None

    class _F:
        def __call__(self):
            return _S()

    monkeypatch.setattr(db_mod, "async_session_factory", _F())
    monkeypatch.setattr(ws_mod, "async_session_factory", _F())

    # Мок _load_credentials чтобы WS-handshake не падал
    from aqr.chat import ws as ws_mod
    from aqr.registry import DecryptedSettings

    fake_creds = DecryptedSettings(
        session_id="integration-test",
        llm_model="claude-3-5-sonnet-20241022",
        llm_api_key="k",
        openai_api_key="k",
        invest_token="t",
        invest_sandbox=True,
    )

    async def fake_load_credentials(session_id):
        return fake_creds

    monkeypatch.setattr(ws_mod, "_load_credentials", fake_load_credentials)


class TestChatPage:
    def test_get_chat_returns_html(self):
        """GET /chat → 200 с HTML-страницей."""
        from fastapi.testclient import TestClient

        from aqr.main import app

        client = TestClient(app)
        r = client.get("/chat")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        body = r.text
        # Smoke-проверки ключевых элементов
        assert "AQR Chat" in body
        assert "<script>" in body
        assert "WebSocket" in body
        assert "/help" in body
        assert "dark" in body.lower() or "#1a1d23" in body

    def test_chat_page_has_login_form(self):
        """Login-форма присутствует в HTML."""
        from fastapi.testclient import TestClient

        from aqr.main import app

        client = TestClient(app)
        body = client.get("/chat").text
        assert 'id="login"' in body
        assert 'id="session-input"' in body
        assert 'id="chat"' in body

    def test_chat_page_has_message_log(self):
        """Область сообщений и input-бар."""
        from fastapi.testclient import TestClient

        from aqr.main import app

        client = TestClient(app)
        body = client.get("/chat").text
        assert 'id="messages"' in body
        assert 'id="input"' in body
        assert 'id="send"' in body

    def test_chat_page_has_help_overlay(self):
        """Help-overlay с описанием slash-команд."""
        from fastapi.testclient import TestClient

        from aqr.main import app

        client = TestClient(app)
        body = client.get("/chat").text
        assert 'id="help-overlay"' in body
        assert "/run" in body
        assert "/history" in body
        assert "/clear" in body
        assert "/exit" in body

    def test_chat_page_has_markdown_renderer(self):
        """Inline JS содержит renderMarkdown()."""
        from fastapi.testclient import TestClient

        from aqr.main import app

        client = TestClient(app)
        body = client.get("/chat").text
        assert "renderMarkdown" in body
        # Покрывает основные markdown-элементы
        assert "<h1>" in body or "h1>" in body
        assert "<strong>" in body or "strong" in body
        assert "<code>" in body or "code" in body


class TestChatNew:
    def test_returns_signed_token_when_auth_enabled(self):
        """С auth — возвращает валидный HMAC-токен."""
        from fastapi.testclient import TestClient

        from aqr.auth import verify_token
        from aqr.main import app

        client = TestClient(app)
        r = client.get("/chat/new", params={"session_id": "alice"})
        assert r.status_code == 200
        data = r.json()
        assert data["session_id"] == "alice"
        assert data["token"] is not None
        assert verify_token(data["token"]) == "alice"

    def test_empty_session_id_rejected(self):
        """Пустой session_id → 422 (валидация Query)."""
        from fastapi.testclient import TestClient

        from aqr.main import app

        client = TestClient(app)
        r = client.get("/chat/new", params={"session_id": ""})
        assert r.status_code == 422

    def test_long_session_id_rejected(self):
        """Слишком длинный session_id (>64) → 422."""
        from fastapi.testclient import TestClient

        from aqr.main import app

        client = TestClient(app)
        r = client.get("/chat/new", params={"session_id": "x" * 65})
        assert r.status_code == 422

    def test_unicode_session_id_accepted(self):
        """Unicode session_id проходит."""
        from fastapi.testclient import TestClient

        from aqr.auth import verify_token
        from aqr.main import app

        client = TestClient(app)
        r = client.get("/chat/new", params={"session_id": "сессия-用户"})
        assert r.status_code == 200
        data = r.json()
        assert data["session_id"] == "сессия-用户"
        assert verify_token(data["token"]) == "сессия-用户"


class TestIntegrationWithWebSocket:
    """Полный flow: получить токен → открыть WS → отправить ping."""

    def test_token_works_with_chat_websocket(self):
        """Токен от /chat/new проходит проверку в /chat/{token}."""
        from fastapi.testclient import TestClient

        from aqr.main import app

        client = TestClient(app)

        # 1. Получаем токен
        token_resp = client.get("/chat/new", params={"session_id": "integration-test"})
        assert token_resp.status_code == 200
        token = token_resp.json()["token"]

        # 2. Подключаемся по WS с этим токеном
        with client.websocket_connect(f"/chat/{token}") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "connected"
            assert msg["session_id"] == "integration-test"

            # 3. Ping → pong
            ws.send_text('{"type": "ping"}')
            resp = ws.receive_json()
            assert resp["type"] == "pong"

    def test_raw_session_id_rejected_when_auth_enabled(self):
        """Без токена WS отвергается (SEC-1)."""
        from fastapi.testclient import TestClient

        from aqr.main import app

        client = TestClient(app)
        with (
            pytest.raises(Exception),
            client.websocket_connect("/chat/raw-session-id") as ws,
        ):
            ws.receive_json()
