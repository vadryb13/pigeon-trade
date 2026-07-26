"""Тесты aqr.agent.context: per-session credentials через ContextVar."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _stable_secret(monkeypatch):
    monkeypatch.setenv("AQR_SESSION_SECRET", "test-secret-padded-to-32-bytes-base64==")


class TestCredentialsContext:
    def test_current_credentials_default_none(self):
        from aqr.agent.context import current_credentials
        assert current_credentials() is None

    def test_set_and_get(self):
        from aqr.agent.context import (
            current_credentials,
            reset_credentials,
            set_credentials,
        )
        from aqr.registry import DecryptedSettings

        creds = DecryptedSettings(
            session_id="alice",
            llm_model="claude-3-5-sonnet-20241022",
            llm_api_key="sk-ant-fake",
            openai_api_key="sk-oai-fake",
            invest_token="t.INVEST_TOKEN_fake",
            invest_sandbox=True,
        )
        token = set_credentials(creds)
        try:
            assert current_credentials() is creds
            assert current_credentials().llm_model == "claude-3-5-sonnet-20241022"
            assert current_credentials().invest_sandbox is True
        finally:
            reset_credentials(token)

        assert current_credentials() is None

    def test_nested_set_resets_properly(self):
        """set вложенный — reset восстанавливает внешнее значение."""
        from aqr.agent.context import (
            current_credentials,
            reset_credentials,
            set_credentials,
        )
        from aqr.registry import DecryptedSettings

        outer = DecryptedSettings(
            session_id="outer",
            llm_model="gpt-4o-mini",
            llm_api_key="k1",
            openai_api_key="k2",
            invest_token="t1",
            invest_sandbox=False,
        )
        inner = DecryptedSettings(
            session_id="inner",
            llm_model="claude-3-5-sonnet-20241022",
            llm_api_key="k3",
            openai_api_key="k4",
            invest_token="t2",
            invest_sandbox=True,
        )

        outer_token = set_credentials(outer)
        try:
            assert current_credentials().session_id == "outer"
            inner_token = set_credentials(inner)
            try:
                assert current_credentials().session_id == "inner"
            finally:
                reset_credentials(inner_token)
            # После reset inner → обратно outer
            assert current_credentials().session_id == "outer"
        finally:
            reset_credentials(outer_token)

        assert current_credentials() is None

    async def test_credentials_visible_in_concurrent_tasks(self):
        """ContextVar изолирован между asyncio-задачами (важно для multi-session)."""
        import asyncio

        from aqr.agent.context import (
            current_credentials,
            reset_credentials,
            set_credentials,
        )
        from aqr.registry import DecryptedSettings

        results: dict[str, str] = {}

        async def worker(name: str, model: str):
            creds = DecryptedSettings(
                session_id=name,
                llm_model=model,
                llm_api_key="k",
                openai_api_key="k",
                invest_token="t",
                invest_sandbox=True,
            )
            token = set_credentials(creds)
            try:
                await asyncio.sleep(0.01)
                results[name] = current_credentials().llm_model
            finally:
                reset_credentials(token)

        await asyncio.gather(
            worker("alice", "claude-3-5-sonnet-20241022"),
            worker("bob", "gpt-4o-mini"),
        )
        assert results["alice"] == "claude-3-5-sonnet-20241022"
        assert results["bob"] == "gpt-4o-mini"
