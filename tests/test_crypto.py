"""Тесты для aqr.crypto — Fernet-шифрование per-session credentials."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _set_secret(monkeypatch):
    monkeypatch.setenv("AQR_SESSION_SECRET", "x" * 32)


class TestEncryptDecrypt:
    def test_roundtrip(self):
        from aqr.crypto import decrypt_str, encrypt_str

        plain = "sk-ant-test-key-12345"
        token = encrypt_str(plain)
        assert token != plain
        assert decrypt_str(token) == plain

    def test_same_plaintext_different_tokens(self):
        """Fernet использует random IV — каждый encrypt даёт уникальный токен."""
        from aqr.crypto import encrypt_str

        plain = "sk-ant-test"
        t1 = encrypt_str(plain)
        t2 = encrypt_str(plain)
        assert t1 != t2

    def test_unicode_roundtrip(self):
        from aqr.crypto import decrypt_str, encrypt_str

        plain = "Привет мир 🌍"
        assert decrypt_str(encrypt_str(plain)) == plain


class TestSecretRotation:
    def test_wrong_secret_raises(self, monkeypatch):
        """При смене AQR_SESSION_SECRET старый токен не расшифровывается."""
        from aqr.crypto import encrypt_str

        # Encrypt с одним secret
        token = encrypt_str("secret")

        # Меняем secret
        monkeypatch.setenv("AQR_SESSION_SECRET", "y" * 32)

        from aqr.crypto import decrypt_str

        with pytest.raises(RuntimeError, match="AQR_SESSION_SECRET may have been rotated"):
            decrypt_str(token)


class TestMissingSecret:
    def test_missing_secret_raises(self, monkeypatch):
        monkeypatch.delenv("AQR_SESSION_SECRET", raising=False)
        from aqr.crypto import encrypt_str

        with pytest.raises(RuntimeError, match="AQR_SESSION_SECRET is required"):
            encrypt_str("anything")

    def test_short_secret_raises(self, monkeypatch):
        monkeypatch.setenv("AQR_SESSION_SECRET", "short")
        from aqr.crypto import encrypt_str

        with pytest.raises(RuntimeError, match="≥32 chars"):
            encrypt_str("anything")
