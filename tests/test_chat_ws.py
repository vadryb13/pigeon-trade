"""Тесты WebSocket-чата через FastAPI TestClient."""
from __future__ import annotations

import json

# SEC-1: для тестов отключаем обязательную проверку токена.
# В реальном prod auth включён через AQR_REQUIRE_WS_AUTH=1 (дефолт).
# Тесты, которым нужен реальный auth flow, используют sign_session() ниже.
import os

import pytest

os.environ["AQR_REQUIRE_WS_AUTH"] = "0"

# ── Mocks для БД ────────────────────────────────────────────────

class _FakeSession:
    """Заглушка AsyncSession — поддерживает async context manager."""
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def commit(self):
        return None

    async def flush(self):
        return None

    async def get(self, *args, **kw):
        return None

    async def execute(self, *args, **kw):
        class _R:
            def scalars(self):
                return self
            def all(self):
                return []
            def scalar(self):
                return None
        return _R()

    def add(self, obj):
        return None

    async def delete(self, *args, **kw):
        return None


class _FakeFactory:
    def __call__(self):
        return _FakeSession()


@pytest.fixture
def mock_db(monkeypatch):
    """Мок _async_session_factory на уровне aqr.db и aqr.chat.ws."""
    from aqr import db as db_mod
    from aqr.chat import ws as ws_mod

    factory = _FakeFactory()
    monkeypatch.setattr(db_mod, "_async_session_factory", factory)
    monkeypatch.setattr(ws_mod, "_async_session_factory", factory)
    return factory


# ── Mocks для агента ─────────────────────────────────────────────

@pytest.fixture
def mock_agent(monkeypatch):
    """Мок графа — astream возвращает финальное состояние, run_agent не вызывается."""
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

    # Патчим на обоих уровнях: и в aqr.agent.graph, и в aqr.chat.ws
    monkeypatch.setattr("aqr.agent.graph.get_agent", fake_get_agent)
    monkeypatch.setattr("aqr.chat.ws.get_agent", fake_get_agent)
    return _FakeAgent


# ── WS тесты ─────────────────────────────────────────────────────

class TestChatWS:
    def test_ws_connect_receives_connected_history(self, mock_db, mock_agent):
        """При connect клиент получает 'connected' и (опц.) 'history'."""
        from fastapi.testclient import TestClient

        from aqr.main import app

        client = TestClient(app)
        with client.websocket_connect("/chat/test-session-1") as ws:
            msg1 = ws.receive_json()
            assert msg1["type"] == "connected"
            assert msg1["session_id"] == "test-session-1"

    def test_ws_ping_returns_pong(self, mock_db, mock_agent):
        """keepalive: ping → pong."""
        from fastapi.testclient import TestClient

        from aqr.main import app

        client = TestClient(app)
        with client.websocket_connect("/chat/test-ping") as ws:
            ws.receive_json()  # connected
            ws.send_text(json.dumps({"type": "ping"}))
            resp = ws.receive_json()
            assert resp["type"] == "pong"

    def test_ws_unknown_type_returns_error(self, mock_db, mock_agent):
        """Неизвестный type → error-сообщение (без падения)."""
        from fastapi.testclient import TestClient

        from aqr.main import app

        client = TestClient(app)
        with client.websocket_connect("/chat/test-unknown") as ws:
            ws.receive_json()  # connected
            ws.send_text(json.dumps({"type": "spam"}))
            resp = ws.receive_json()
            assert resp["type"] == "error"
            assert "spam" in resp["message"]

    def test_ws_invalid_json_returns_error(self, mock_db, mock_agent):
        """Битый JSON → error-сообщение."""
        from fastapi.testclient import TestClient

        from aqr.main import app

        client = TestClient(app)
        with client.websocket_connect("/chat/test-invalid") as ws:
            ws.receive_json()  # connected
            ws.send_text("{not json")
            resp = ws.receive_json()
            assert resp["type"] == "error"
            assert "Invalid JSON" in resp["message"]

    def test_ws_message_triggers_agent_and_done(self, mock_db, mock_agent):
        """user-message → agent → progress → user_echo → done."""
        from fastapi.testclient import TestClient

        from aqr.main import app

        client = TestClient(app)
        with client.websocket_connect("/chat/test-msg") as ws:
            ws.receive_json()  # connected

            ws.send_text(json.dumps({
                "type": "message",
                "content": "проверь momentum на Сбере",
            }))

            kinds = []
            while True:
                msg = ws.receive_json()
                kinds.append(msg["type"])
                if msg["type"] in ("done", "error"):
                    break

            # Должны получить: user_echo, progress, ..., done
            assert "user_echo" in kinds
            assert "done" in kinds
            # Финальное сообщение содержит нарратив
            done = next(m for m in [msg] if m["type"] == "done")
            assert "narrative" in done
            assert "Тестовый отчёт" in done["narrative"]

    def test_ws_message_empty_content_ignored(self, mock_db, mock_agent):
        """Пустой content → no-op (агент не запускается)."""
        from fastapi.testclient import TestClient

        from aqr.main import app

        client = TestClient(app)
        with client.websocket_connect("/chat/test-empty") as ws:
            ws.receive_json()  # connected
            ws.send_text(json.dumps({"type": "message", "content": "  "}))

            # Не должно быть user_echo или progress — клиент просто ждёт дальше.
            # Подтверждаем: connection всё ещё открыт, можно отправить ping.
            ws.send_text(json.dumps({"type": "ping"}))
            resp = ws.receive_json()
            assert resp["type"] == "pong"

    def test_ws_two_sessions_isolated(self, mock_db, mock_agent):
        """Две WS в разных session_id независимы (получают разные session_id в connected)."""
        from fastapi.testclient import TestClient

        from aqr.main import app

        client = TestClient(app)
        with (
            client.websocket_connect("/chat/session-A") as ws_a,
            client.websocket_connect("/chat/session-B") as ws_b,
        ):
            msg_a = ws_a.receive_json()
            msg_b = ws_b.receive_json()
            assert msg_a["session_id"] == "session-A"
            assert msg_b["session_id"] == "session-B"
            assert msg_a["session_id"] != msg_b["session_id"]

    def test_resume_returns_history_message(self, mock_db, mock_agent, monkeypatch):
        """resume → history-сообщение (даже если БД пуста)."""
        from fastapi.testclient import TestClient

        from aqr.main import app

        client = TestClient(app)
        with client.websocket_connect("/chat/test-resume") as ws:
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
        """save_chat_message вызывается с правильными аргументами.

        Не использует mock_db фикстуру — тест полностью self-contained,
        чтобы избежать cross-test pollution с test_api_routes.
        """
        from aqr import db as db_mod
        from aqr.registry.store import RegistryStore

        # Свой session factory
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

        monkeypatch.setattr(db_mod, "_async_session_factory", _F())

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
        """Если БД недоступна — list_chat_history возвращает []."""
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
        """Если БД упала при save — WS не падает (logged warning)."""
        from aqr.chat import ws as ws_mod
        from aqr.registry.store import RegistryStore

        async def broken_save(self, *args, **kw):
            raise RuntimeError("DB down")

        async def run_save():
            RegistryStore.save_chat_message = broken_save
            await ws_mod._save_history("test", "user", "hi")

        # Должно выполниться без exceptions
        await run_save()
