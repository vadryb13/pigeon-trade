"""Тесты для aqr/auth.py — HMAC-подпись session_id (SEC-1)."""
from __future__ import annotations

import os

import pytest


@pytest.fixture
def _stable_secret(monkeypatch):
    """Зафиксировать AQR_SESSION_SECRET для воспроизводимости."""
    monkeypatch.setenv("AQR_SESSION_SECRET", "test-secret-32-bytes-base64==")


class TestSignAndVerify:
    def test_round_trip(self, _stable_secret):
        from aqr.auth import sign_session, verify_token

        token = sign_session("alice-session")
        assert verify_token(token) == "alice-session"

    def test_default_session(self, _stable_secret):
        from aqr.auth import issue_default_token, verify_token

        token = issue_default_token()
        assert verify_token(token) == "default"

    def test_unicode_session_id(self, _stable_secret):
        from aqr.auth import sign_session, verify_token

        token = sign_session("сессия-用户-123")
        assert verify_token(token) == "сессия-用户-123"

    def test_different_sessions_different_tokens(self, _stable_secret):
        from aqr.auth import sign_session

        t1 = sign_session("a")
        t2 = sign_session("b")
        assert t1 != t2


class TestVerifyRejectsBadTokens:
    def test_empty_token(self, _stable_secret):
        from aqr.auth import verify_token

        assert verify_token("") is None

    def test_no_prefix(self, _stable_secret):
        from aqr.auth import verify_token

        assert verify_token("v1.something") is None
        assert verify_token("garbage.token") is None

    def test_wrong_signature(self, _stable_secret):
        from aqr.auth import sign_session, verify_token

        token = sign_session("alice")
        # Портим подпись
        parts = token.split(".")
        parts[-1] = "AAAA"  # невалидный base64 подписи
        tampered = ".".join(parts)
        assert verify_token(tampered) is None

    def test_signature_with_wrong_secret(self, _stable_secret):
        from aqr.auth import sign_session, verify_token

        token = sign_session("alice")
        # Меняем секрет — старая подпись не должна пройти
        os.environ["AQR_SESSION_SECRET"] = "different-secret-32-bytes-base64"
        assert verify_token(token) is None


class TestEphemeralSecretFallback:
    def test_no_env_uses_ephemeral(self, monkeypatch):
        """Без AQR_SESSION_SECRET — токен создаётся через ephemeral-ключ."""
        monkeypatch.delenv("AQR_SESSION_SECRET", raising=False)
        import aqr.auth as auth_mod

        # Сбрасываем кеш ephemeral-секрета в модуле
        monkeypatch.setattr(auth_mod, "_EPHEMERAL_SECRET", None)
        # В рамках одного процесса ephemeral стабилен → verify проходит
        token = auth_mod.sign_session("s1")
        assert auth_mod.verify_token(token) == "s1"

    def test_different_secrets_dont_cross_verify(self, monkeypatch):
        """Разные секреты (ephemeral vs env) не пересекаются."""
        import aqr.auth as auth_mod

        monkeypatch.setattr(auth_mod, "_EPHEMERAL_SECRET", None)
        monkeypatch.delenv("AQR_SESSION_SECRET", raising=False)
        token_no_env = auth_mod.sign_session("alice")

        monkeypatch.setenv("AQR_SESSION_SECRET", "new-fixed-secret-32-bytes==")
        assert auth_mod.verify_token(token_no_env) is None
