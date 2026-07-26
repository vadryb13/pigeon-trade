"""Тесты для aqr.llm_env.llm_credentials_env()."""
from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _stable_secret(monkeypatch):
    monkeypatch.setenv("AQR_SESSION_SECRET", "test-secret-padded-to-32-bytes-base64==")


def _creds(model: str, key: str = "test-key"):
    from aqr.registry import DecryptedSettings
    return DecryptedSettings(
        session_id="alice",
        llm_model=model,
        llm_api_key=key,
        openai_api_key="oai",
        invest_token="t",
        invest_sandbox=True,
    )


class TestLlmCredentialsEnv:
    def test_anthropic_model_sets_anthropic_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        from aqr.llm_env import llm_credentials_env

        with llm_credentials_env(_creds("claude-3-5-sonnet-20241022", "sk-ant-x")):
            assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-x"
            assert "OPENAI_API_KEY" not in os.environ
        assert "ANTHROPIC_API_KEY" not in os.environ

    def test_openai_model_sets_openai_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        from aqr.llm_env import llm_credentials_env

        with llm_credentials_env(_creds("gpt-4o-mini", "sk-oai-x")):
            assert os.environ["OPENAI_API_KEY"] == "sk-oai-x"
            assert "ANTHROPIC_API_KEY" not in os.environ
        assert "OPENAI_API_KEY" not in os.environ

    def test_gigachat_model_sets_gigachat(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GIGACHAT_CREDENTIALS", raising=False)

        from aqr.llm_env import llm_credentials_env

        with llm_credentials_env(_creds("gigachat/GigaChat-Pro", "giga-cred")):
            assert os.environ["GIGACHAT_CREDENTIALS"] == "giga-cred"
            assert "ANTHROPIC_API_KEY" not in os.environ
            assert "OPENAI_API_KEY" not in os.environ

    def test_restores_previous_env_on_exit(self, monkeypatch):
        """До входа в контекст env установлен — после выхода восстанавливается."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "PREVIOUS_KEY")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        from aqr.llm_env import llm_credentials_env

        with llm_credentials_env(_creds("gpt-4o-mini", "sk-oai-new")):
            assert os.environ["OPENAI_API_KEY"] == "sk-oai-new"
            # Внутри контекста ANTHROPIC_API_KEY удалён
            assert "ANTHROPIC_API_KEY" not in os.environ

        # После выхода — обратно PREVIOUS_KEY
        assert os.environ["ANTHROPIC_API_KEY"] == "PREVIOUS_KEY"
        # OPENAI_API_KEY не было до входа — должно быть удалено
        assert "OPENAI_API_KEY" not in os.environ

    def test_restores_on_exception(self, monkeypatch):
        """Даже если вложенный код бросил — env восстанавливается."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ORIGINAL")

        from aqr.llm_env import llm_credentials_env

        with pytest.raises(RuntimeError), llm_credentials_env(
            _creds("gpt-4o-mini", "sk-oai-x")
        ):
            raise RuntimeError("boom")

        assert os.environ["ANTHROPIC_API_KEY"] == "ORIGINAL"
        assert "OPENAI_API_KEY" not in os.environ
