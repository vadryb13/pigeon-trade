"""Fernet-шифрование per-session credentials.

Ключ — производный от AQR_SESSION_SECRET через HKDF-SHA256:
    IKM  = AQR_SESSION_SECRET (env)
    salt = b"aqr-session-credentials"
    info = b"aqr/fernet/v1"
    key  = HKDF(...).derive(IKM)[:32] → Fernet

При ротации AQR_SESSION_SECRET все ранее зашифрованные credentials
становятся нечитаемыми (InvalidToken). Сессии нужно пересохранить —
это ожидаемое поведение, считается инвалидацией всех настроек.
"""
from __future__ import annotations

import base64
import os

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

_SALT = b"aqr-session-credentials"
_INFO = b"aqr/fernet/v1"
_KEY_LENGTH = 32


def _master_secret() -> bytes:
    secret = os.environ.get("AQR_SESSION_SECRET")
    if not secret:
        raise RuntimeError("AQR_SESSION_SECRET is required")
    if len(secret) < 32:
        raise RuntimeError(
            f"AQR_SESSION_SECRET must be ≥32 chars (got {len(secret)})"
        )
    return secret.encode("utf-8")


def _fernet() -> Fernet:
    derived = HKDF(
        algorithm=hashes.SHA256(),
        length=_KEY_LENGTH,
        salt=_SALT,
        info=_INFO,
    ).derive(_master_secret())
    return Fernet(base64.urlsafe_b64encode(derived))


def encrypt_str(plain: str) -> str:
    """Зашифровать строку → Fernet-токен (urlsafe base64)."""
    return _fernet().encrypt(plain.encode("utf-8")).decode("ascii")


def decrypt_str(token: str) -> str:
    """Расшифровать Fernet-токен → строка. На неверном ключе — InvalidToken."""
    try:
        return _fernet().decrypt(token.encode("ascii")).decode("utf-8")
    except InvalidToken as e:
        raise RuntimeError(
            "Cannot decrypt credentials — AQR_SESSION_SECRET may have been rotated. "
            "Re-save session settings."
        ) from e
