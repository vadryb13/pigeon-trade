"""Тесты WebSocket-чата через FastAPI TestClient (HMAC auth обязателен)."""
from __future__ import annotations

import json

import pytest

from conftest import FakeSession


# ── Mocks для БД ────────────────────────────────────────────────

@pytest.fixture
def mock_db(monkeypatch):
    """Мок async_session_factory на уровне aqr.db и aqr.chat.ws."""
    from aqr import session as db_mod
    from aqr.chat import ws as ws_mod

    factory = lambda: FakeSession()
    monkeypatch.setattr(db_mod, "async_session_factory", factory)
    monkeypatch.setattr(ws_mod, "async_session_factory", factory)
    return factory


@pytest.fixture
def mock_credentials(monkeypatch):
    """Мок _load_credentials → возвращает фейковые credentials сессии."""
    from aqr.chat import ws as ws_mod
    from aqr.registry import DecryptedSettings

    fake_creds = DecryptedSettings(
        session_id="alice",
        llm_model="claude-3-5-sonnet-20241022",
        llm_api_key="sk-ant-fake",
        openai_api_key="sk-oai-fake",
        invest_token="t.INVEST_TOKEN_fake",
        invest_sandbox=True,
    )

    async def fake_load_credentials(session_id: str):
        return fake_creds

    monkeypatch.setattr(ws_mod, "_load_credentials", fake_load_credentials)
    return fake_creds


@pytest.fixture
def mock_no_credentials(monkeypatch):
    """Мок _load_credentials → возвращает None (settings не настроены)."""
    from aqr.chat import ws as ws_mod

    async def fake_load_credentials(session_id: str):
        return None

    monkeypatch.setattr(ws_mod, "_load_credentials", fake_load_credentials)


# ── Mocks для агента ─────────────────────────────────────────────

@pytest.fixture
def mock_agent(monkeypatch):
    """Мок графа — astream возвращает финальное состояние."""
    fake_state = {
        "step": "plan",
        "plan": {"tickers": ["SBER"], "hypothesis_families": ["momentum"]},
        "results": [
            {"dsr_verdict": "significant", "sharpe": 1.2, "dsr": 0.96},
        ],
        "narrative": "Тестовый отчёт по momentum.",
        "n_tested": 1, "n_survived": 1,
        "messages": [
            {"role": "user", "content": "проверь momentum на Сбере"},
            {"role": "assistant", "content": "Тестовый отчёт по momentum."},
        ],
    }

    class _FakeAgent:
        async def astream(self, state, stream_mode=None):
            yield fake_state

        async def ainvoke(self, state):
            return fake_state

    def fake_get_agent():
        return _FakeAgent()

    monkeypatch.setattr("aqr.graph.graph.get_agent", fake_get_agent)
    monkeypatch.setattr("aqr.chat.ws.get_agent", fake_get_agent)
    return _FakeAgent


# ── WS тесты ─────────────────────────────────────────────────────

def _sign(session_id: str) -> str:
    from aqr.auth import sign_session
    return sign_session(session_id)


class TestChatWS:
    def test_ws_connect_receives_connected(
        self, mock_db, mock_credentials, mock_agent
    ):
        from fastapi.testclient import TestClient

        from aqr.main import app

        client = TestClient(app)
        token = _sign("test-session-1")
        with client.websocket_connect(f"/chat/{token}") as ws:
            msg1 = ws.receive_json()
            assert msg1["type"] == "connected"
            assert msg1["session_id"] == "test-session-1"
            assert msg1["credentials_configured"] is True

    def test_ws_invalid_token_closed(
        self, mock_db, mock_credentials, mock_agent
    ):
        """Без валидного HMAC-токена WS закрывается (1008)."""
        from fastapi.testclient import TestClient

        from aqr.main import app

        client = TestClient(app)
        with pytest.raises(Exception), client.websocket_connect("/chat/raw-no-hmac") as ws:
            ws.receive_json()

    def test_ws_ping_returns_pong(self, mock_db, mock_credentials, mock_agent):
        from fastapi.testclient import TestClient

        from aqr.main import app

        client = TestClient(app)
        token = _sign("test-ping")
        with client.websocket_connect(f"/chat/{token}") as ws:
            ws.receive_json()  # connected
            ws.send_text(json.dumps({"type": "ping"}))
            resp = ws.receive_json()
            assert resp["type"] == "pong"

    def test_ws_unknown_type_returns_error(
        self, mock_db, mock_credentials, mock_agent
    ):
        from fastapi.testclient import TestClient

        from aqr.main import app

        client = TestClient(app)
        token = _sign("test-unknown")
        with client.websocket_connect(f"/chat/{token}") as ws:
            ws.receive_json()  # connected
            ws.send_text(json.dumps({"type": "spam"}))
            resp = ws.receive_json()
            assert resp["type"] == "error"
            assert "spam" in resp["message"]

    def test_ws_invalid_json_returns_error(
        self, mock_db, mock_credentials, mock_agent
    ):
        from fastapi.testclient import TestClient

        from aqr.main import app

        client = TestClient(app)
        token = _sign("test-invalid")
        with client.websocket_connect(f"/chat/{token}") as ws:
            ws.receive_json()  # connected
            ws.send_text("{not json")
            resp = ws.receive_json()
            assert resp["type"] == "error"
            assert "Invalid JSON" in resp["message"]

    def test_ws_message_triggers_agent_and_done(
        self, mock_db, mock_credentials, mock_agent
    ):
        from fastapi.testclient import TestClient

        from aqr.main import app

        client = TestClient(app)
        token = _sign("test-msg")
        with client.websocket_connect(f"/chat/{token}") as ws:
            ws.receive_json()  # connected

            ws.send_text(json.dumps({
                "type": "message",
                "content": "проверь momentum на Сбере",
            }))

            kinds = []
            final_msg = None
            while True:
                msg = ws.receive_json()
                kinds.append(msg["type"])
                if msg["type"] == "done":
                    final_msg = msg
                    break
                if msg["type"] == "error":
                    break

            assert "user_echo" in kinds
            assert "done" in kinds
            assert final_msg is not None
            assert "narrative" in final_msg
            assert "Тестовый отчёт" in final_msg["narrative"]

    def test_ws_message_without_credentials_returns_error(
        self, mock_db, mock_no_credentials, mock_agent
    ):
        """Без настроенных credentials → message → error с ссылкой на /settings."""
        from fastapi.testclient import TestClient

        from aqr.main import app

        client = TestClient(app)
        token = _sign("no-settings-session")
        with client.websocket_connect(f"/chat/{token}") as ws:
            connected = ws.receive_json()
            assert connected["credentials_configured"] is False

            settings_error = ws.receive_json()
            assert settings_error["type"] == "error"
            assert "/settings" in settings_error["message"]

    def test_ws_message_empty_content_ignored(
        self, mock_db, mock_credentials, mock_agent
    ):
        from fastapi.testclient import TestClient

        from aqr.main import app

        client = TestClient(app)
        token = _sign("test-empty")
        with client.websocket_connect(f"/chat/{token}") as ws:
            ws.receive_json()  # connected
            ws.send_text(json.dumps({"type": "message", "content": "  "}))

            ws.send_text(json.dumps({"type": "ping"}))
            resp = ws.receive_json()
            assert resp["type"] == "pong"

    def test_ws_two_sessions_isolated(
        self, mock_db, mock_credentials, mock_agent
    ):
        from fastapi.testclient import TestClient

        from aqr.main import app

        client = TestClient(app)
        with (
            client.websocket_connect(f"/chat/{_sign('session-A')}") as ws_a,
            client.websocket_connect(f"/chat/{_sign('session-B')}") as ws_b,
        ):
            msg_a = ws_a.receive_json()
            msg_b = ws_b.receive_json()
            assert msg_a["session_id"] == "session-A"
            assert msg_b["session_id"] == "session-B"
            assert msg_a["session_id"] != msg_b["session_id"]

    def test_resume_returns_history_message(
        self, mock_db, mock_credentials, mock_agent
    ):
        from fastapi.testclient import TestClient

        from aqr.main import app

        client = TestClient(app)
        token = _sign("test-resume")
        with client.websocket_connect(f"/chat/{token}") as ws:
            ws.receive_json()  # connected
            ws.send_text(json.dumps({"type": "resume"}))
            resp = ws.receive_json()
            assert resp["type"] == "history"
            assert "messages" in resp
            assert isinstance(resp["messages"], list)


# ── Тест на storage-метод RegistryStore (без БД, через мок) ─────

class TestSaveAndListChatHistory:
    """Тесты на RegistryStore.save_chat_message / list_chat_history."""

    @pytest.mark.asyncio
    async def test_save_calls_store_method(self, monkeypatch):
        from aqr import session as db_mod
        from aqr.registry.store import RegistryStore

        class _S:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return None
            async def commit(self):
                pass
            async def flush(self):
                pass

        class _F:
            def __call__(self):
                return _S()

        monkeypatch.setattr(db_mod, "async_session_factory", _F())

        captured = {}

        async def fake_save(_self, session_id, role, content, meta=None):
            captured["args"] = (session_id, role, content, meta)
            return None

        monkeypatch.setattr(
            RegistryStore, "save_chat_message", fake_save,
        )

        from aqr.chat import ws as ws_mod
        await ws_mod._save_history("test-session", "user", "привет", {"k": "v"})
        assert captured["args"] == ("test-session", "user", "привет", {"k": "v"})

    @pytest.mark.asyncio
    async def test_load_returns_empty_list_when_db_fails(self, mock_db, monkeypatch):
        from aqr.chat import ws as ws_mod
        from aqr.registry.store import RegistryStore

        async def broken_list(self, *args, **kw):
            raise RuntimeError("DB down")

        monkeypatch.setattr(
            RegistryStore, "list_chat_history", broken_list,
        )

        result = await ws_mod._load_history("test")
        assert result == []

    @pytest.mark.asyncio
    async def test_save_handles_db_failure_silently(self, mock_db):
        from aqr.chat import ws as ws_mod
        from aqr.registry.store import RegistryStore

        async def broken_save(self, *args, **kw):
            raise RuntimeError("DB down")

        async def run_save():
            RegistryStore.save_chat_message = broken_save
            await ws_mod._save_history("test", "user", "hi")

        await run_save()
