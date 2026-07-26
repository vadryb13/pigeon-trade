"""Тесты для SessionSettings CRUD в RegistryStore.

Проверяют save/load/delete + decrypt_settings round-trip с моком БД.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _set_secret(monkeypatch):
    monkeypatch.setenv("AQR_SESSION_SECRET", "x" * 32)


class _FakeDB:
    """Минимальный мок AsyncSession для тестов SessionSettings."""

    def __init__(self):
        self.added: list = []
        self.deleted: list = []
        self._store: dict[tuple[type, str], object] = {}

    def _pk(self, model, obj):
        return (model, getattr(obj, "id", None) or obj.session_id)

    def add(self, obj):
        self.added.append(obj)
        # SessionSettings → PK = session_id; Session → PK = id
        if hasattr(obj, "session_id") and not hasattr(obj, "id"):
            self._store[(type(obj), obj.session_id)] = obj
        elif hasattr(obj, "id"):
            self._store[(type(obj), obj.id)] = obj

    async def get(self, model, key):
        return self._store.get((model, key))

    async def delete(self, obj):
        self.deleted.append(obj)
        self._store.pop(self._pk(model=type(obj), obj=obj), None)

    async def flush(self):
        return None


class TestSessionSettingsCRUD:
    async def test_save_creates_new_settings(self):
        from aqr.registry import RegistryStore, SessionSettings

        db = _FakeDB()
        store = RegistryStore(db)

        result = await store.save_session_settings(
            session_id="alice",
            llm_model="claude-3-5-sonnet-20241022",
            llm_api_key="sk-ant-fake",
            openai_api_key="sk-oai-fake",
            invest_token="t.INVEST_TOKEN_fake",
            invest_sandbox=True,
        )

        assert isinstance(result, SessionSettings)
        assert result.session_id == "alice"
        assert result.llm_model == "claude-3-5-sonnet-20241022"
        assert result.invest_sandbox is True
        # Ключи зашифрованы в БД (не plaintext)
        assert "sk-ant-fake" not in result.llm_api_key_encrypted
        assert result.llm_api_key_encrypted.startswith("gAAAAA")

    async def test_save_updates_existing_settings(self):
        db = _FakeDB()
        from aqr.registry import RegistryStore

        store = RegistryStore(db)

        s1 = await store.save_session_settings(
            "alice", "claude-3-5-sonnet-20241022",
            "k1", "k2", "t1", True,
        )
        original_updated = s1.updated_at

        # Гарантируем что datetime.now() даст более позднее время
        import asyncio
        await asyncio.sleep(0.01)

        s2 = await store.save_session_settings(
            "alice", "gpt-4o-mini",
            "k3", "k4", "t2", False,
        )
        assert s2 is s1
        assert s2.llm_model == "gpt-4o-mini"
        assert s2.invest_sandbox is False
        assert s2.updated_at > original_updated
        # Только одна SessionSettings запись (Session был отдельно при первом save)
        settings_count = sum(
            1 for k in db._store
            if k[0].__name__ == "SessionSettings"
        )
        assert settings_count == 1

    async def test_load_returns_none_if_not_set(self):
        db = _FakeDB()
        from aqr.registry import RegistryStore

        result = await RegistryStore(db).load_session_settings("nobody")
        assert result is None

    async def test_delete_removes_settings(self):
        db = _FakeDB()
        from aqr.registry import RegistryStore

        store = RegistryStore(db)
        await store.save_session_settings(
            "alice", "claude-3-5-sonnet-20241022", "k1", "k2", "t1",
        )

        await store.delete_session_settings("alice")
        # SessionSettings удалён, Session (auto-created) остаётся
        settings_count = sum(
            1 for k in db._store
            if k[0].__name__ == "SessionSettings"
        )
        assert settings_count == 0
        assert len(db.deleted) == 1

    async def test_delete_noop_if_not_exists(self):
        db = _FakeDB()
        from aqr.registry import RegistryStore

        await RegistryStore(db).delete_session_settings("nobody")


class TestDecryptSettings:
    async def test_roundtrip_via_decrypt(self):
        from aqr.registry import RegistryStore, decrypt_settings

        db = _FakeDB()
        store = RegistryStore(db)
        saved = await store.save_session_settings(
            "alice", "claude-3-5-sonnet-20241022",
            "sk-ant-abc", "sk-oai-xyz", "t.INVEST_TOKEN_123",
            invest_sandbox=False,
        )

        plaintext = decrypt_settings(saved)
        assert plaintext.session_id == "alice"
        assert plaintext.llm_model == "claude-3-5-sonnet-20241022"
        assert plaintext.llm_api_key == "sk-ant-abc"
        assert plaintext.openai_api_key == "sk-oai-xyz"
        assert plaintext.invest_token == "t.INVEST_TOKEN_123"
        assert plaintext.invest_sandbox is False

    async def test_decrypt_with_wrong_secret_fails(self, monkeypatch):
        from aqr.registry import RegistryStore, decrypt_settings

        db = _FakeDB()
        store = RegistryStore(db)
        saved = await store.save_session_settings(
            "alice", "claude-3-5-sonnet-20241022", "k1", "k2", "t1",
        )

        monkeypatch.setenv("AQR_SESSION_SECRET", "y" * 32)

        with pytest.raises(RuntimeError, match="may have been rotated"):
            decrypt_settings(saved)
