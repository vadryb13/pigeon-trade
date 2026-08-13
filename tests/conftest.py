"""Shared mock classes and fixtures for all tests."""
from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

# ── Fake DB session ──────────────────────────────────────────────


class FakeSession:
    """Standalone mock for an async SQLAlchemy session.

    Supports basic CRUD, commit, flush, and context manager.
    """

    def __init__(self):
        self._rows: list = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return None

    async def execute(self, *a, **kw):
        return _FakeResult()

    async def get(self, *a, **kw):
        return None

    async def commit(self):
        return None

    async def rollback(self):
        return None

    async def flush(self):
        return None

    def add(self, *a, **kw):
        return None


class BrokenFactory:
    """Factory that simulates a broken DB — raises on first call."""

    def __call__(self):
        raise RuntimeError("DB down")


class _FakeResult:
    """Fake SQLAlchemy result — returns empty scalars/all."""

    def scalars(self):
        return self

    def all(self):
        return []

    def scalar(self):
        return None


# ── T-Invest adapter mock ────────────────────────────────────────


class FakeAdapter:
    """Mock TInvestAdapter.candles returns synthetic DataFrame."""

    def __init__(self, *a, **kw):
        pass

    async def candles(self, ticker="SBER", *a, **kw):
        import pandas as pd

        rng = pd.date_range("2023-01-02", periods=500, freq="B")
        px = [100 + i * 0.05 for i in range(500)]
        return pd.DataFrame(
            {"open": px, "high": px, "low": px, "close": px, "volume": [0] * 500},
            index=rng,
        )

    async def _resolve_figi(self, ticker: str = "SBER") -> str:
        return "BBG004730N88"


class CountingAdapter(FakeAdapter):
    """FakeAdapter that counts calls for cache-hit verification."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.call_count = 0

    async def candles(self, ticker="SBER", *a, **kw):
        self.call_count += 1
        return await super().candles(ticker, *a, **kw)


# ── OpenAI mock for embeddings ───────────────────────────────────


class FakeEmbeddingsAPI:
    """Mock openai.AsyncClient().embeddings.create() → deterministic vector per text."""

    async def create(self, *, model, input, **kw):
        if isinstance(input, str):
            texts = [input]
        else:
            texts = list(input)
        data = []
        for t in texts:
            vec = _embedding_for_text(t)
            data.append(MagicMock(embedding=vec))
        return MagicMock(data=data)


def _embedding_for_text(text: str) -> list[float]:
    """Deterministic embedding where conceptually similar texts are closer.

    Two texts about 'momentum on SBER' will have positive vec[0] and vec[1],
    while a text about 'currency pairs EUR USD' will have those at zero and
    vec[2] positive — this makes cosine similarity higher for similar texts.
    """
    vec = [0.0] * 768
    if "momentum" in text or "SMA" in text:
        vec[0] = 1.0
    if "SBER" in text:
        vec[1] = 0.8
    if "GAZP" in text:
        vec[1] = -0.4
    if "EUR" in text or "currency" in text or "pair" in text:
        vec[2] = 1.0
    # Normalize to unit length
    mag = sum(v * v for v in vec) ** 0.5
    if mag:
        vec = [v / mag for v in vec]
    return vec


class FakeAsyncOpenAI:
    """Mock openai.AsyncOpenAI client."""

    def __init__(self, **kw):
        self.embeddings = FakeEmbeddingsAPI()


def fake_openai_module():
    """Create a fake `openai` module with AsyncOpenAI class."""
    mod = types.ModuleType("openai")
    mod.AsyncOpenAI = FakeAsyncOpenAI
    return mod


# ── litellm fixture ──────────────────────────────────────────────


@pytest.fixture
def fake_litellm(monkeypatch):
    """Mock litellm.acompletion — отвечает JSON в зависимости от system prompt.

    Использование:
        fake_litellm('{"action": "plan"}')
    """

    def _install(response_json: str) -> AsyncMock:
        fake_resp = MagicMock()
        fake_resp.choices = [MagicMock()]
        fake_resp.choices[0].message.content = response_json

        fake_module = types.ModuleType("litellm")
        fake_module.acompletion = AsyncMock(return_value=fake_resp)
        monkeypatch.setitem(sys.modules, "litellm", fake_module)
        return fake_module.acompletion

    return _install


# ── Credentials fixture ──────────────────────────────────────────


@pytest.fixture
def with_credentials():
    """Устанавливает credentials в ContextVar, очищает на teardown."""
    from aqr.graph.context import reset_credentials, set_credentials
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
    yield creds
    reset_credentials(token)


# ── Stable secret (autouse — все тесты его переопределяют при необходимости) ──


@pytest.fixture(autouse=True)
def stable_secret(monkeypatch):
    """Set AQR_SESSION_SECRET to a known value."""
    monkeypatch.setenv("AQR_SESSION_SECRET", "test-secret-padded-to-32-bytes-base64==")
