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
    from aqr.chat import web as web_mod
    monkeypatch.setattr(web_mod, "async_session_factory", _F())
    from aqr.auth import verify_token

    async def fake_verify_token_async(token, _factory):
        return verify_token(token)

    monkeypatch.setattr(ws_mod, "verify_token_async", fake_verify_token_async)

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

    def test_chat_page_creates_server_session(self):
        """UI не просит пользователя задавать предсказуемый session_id."""
        from fastapi.testclient import TestClient

        from aqr.main import app

        client = TestClient(app)
        body = client.get("/chat").text
        assert 'id="login"' in body
        assert "Новая защищённая сессия" in body
        assert "fetch('/chat/new', { method: 'POST' })" in body
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
    def test_creates_signed_opaque_session(self):
        """Сервер возвращает HMAC токен, не принимая client-controlled ID."""
        from fastapi.testclient import TestClient

        from aqr.auth import verify_token
        from aqr.main import app

        client = TestClient(app)
        r = client.post("/chat/new")
        assert r.status_code == 200
        data = r.json()
        assert len(data["session_id"]) == 36
        assert data["token"] is not None
        assert verify_token(data["token"]) == data["session_id"]
        assert "HttpOnly" in r.headers["set-cookie"]

    def test_client_session_id_is_not_accepted(self):
        """Query-параметр не влияет на выданную сессию."""
        from fastapi.testclient import TestClient

        from aqr.main import app

        client = TestClient(app)
        r = client.post("/chat/new?session_id=alice")
        assert r.status_code == 200
        data = r.json()
        assert data["session_id"] != "alice"


class TestIntegrationWithWebSocket:
    """Полный flow: получить токен → открыть WS → отправить ping."""

    def test_token_works_with_chat_websocket(self):
        """Токен от /chat/new проходит проверку в /chat/{token}."""
        from fastapi.testclient import TestClient

        from aqr.main import app

        client = TestClient(app)

        # 1. Получаем токен
        token_resp = client.post("/chat/new")
        assert token_resp.status_code == 200
        token = token_resp.json()["token"]

        # 2. Подключаемся по WS с этим токеном
        with client.websocket_connect(f"/chat/{token}") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "connected"
            assert msg["session_id"] == token_resp.json()["session_id"]

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
