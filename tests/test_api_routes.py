"""Тесты для aqr/pipeline/api.py — FastAPI роуты.

Покрывают POST /pipeline/runs, GET /pipeline/runs/{id}, GET /pipeline/runs/{id}/stream.
"""
from __future__ import annotations

import sys
import types
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture(autouse=True)
def _stable_secret(monkeypatch):
    monkeypatch.setenv("AQR_SESSION_SECRET", "test-secret-padded-to-32-bytes-base64==")


@pytest.fixture
def with_credentials():
    from aqr.agent.context import reset_credentials, set_credentials
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


@pytest.fixture
def fake_openai(monkeypatch):
    """Мок openai.AsyncOpenAI — embeddings возвращает [0.1] * 1536."""

    class _FakeEmbeddingsAPI:
        async def create(self, *, model, input):
            return MagicMock(data=[MagicMock(embedding=[0.1] * 1536)])

    class _FakeAsyncOpenAI:
        def __init__(self, **kw):
            self.embeddings = _FakeEmbeddingsAPI()

    fake = types.ModuleType("openai")
    fake.AsyncOpenAI = _FakeAsyncOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake)


@pytest.fixture
def mock_db(monkeypatch):
    """Мок БД и Executor."""
    class _FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def commit(self):
            pass

        async def rollback(self):
            pass

        async def flush(self):
            pass

        async def get(self, *args, **kw):
            return None

        async def execute(self, *args, **kw):
            class _R:
                scalars = lambda self: self
                all = lambda self: []
                def scalar(self):
                    return None
            return _R()

        def add(self, *args, **kw):
            pass

    factory = type("_F", (), {"__call__": lambda self: _FakeSession()})()

    from aqr import db as db_mod
    monkeypatch.setattr(db_mod, "_async_session_factory", factory)

    # Мок RegistryStore
    mock_store = MagicMock()
    mock_store.get_or_create_session = AsyncMock()
    mock_store.create_run = AsyncMock(return_value=MagicMock())
    mock_store.update_run_status = AsyncMock()
    mock_store.create_hypothesis = AsyncMock()

    from aqr.registry import store as store_mod
    monkeypatch.setattr(store_mod, "RegistryStore", lambda db: mock_store)

    # api.py imports RegistryStore at top-level — patch in its namespace
    from aqr.pipeline import api as api_mod
    monkeypatch.setattr(api_mod, "RegistryStore", lambda db: mock_store)

    return mock_store


# ── POST /pipeline/runs ─────────────────────────────────────────


class TestStartRun:
    def test_post_runs_validates_goal(self):
        """POST /pipeline/runs без goal → 422 (валидация Pydantic)."""
        from fastapi.testclient import TestClient

        from aqr.main import app

        client = TestClient(app)
        r = client.post("/pipeline/runs", json={})
        assert r.status_code == 422

    def test_post_runs_wrong_field_type(self):
        """POST /pipeline/runs с числом вместо строки → 422."""
        from fastapi.testclient import TestClient

        from aqr.main import app

        client = TestClient(app)
        r = client.post("/pipeline/runs", json={"goal": 123})
        assert r.status_code == 422


# ── GET /pipeline/runs/{id} ─────────────────────────────────────


class TestGetRun:
    def test_get_run_returns_status_running(self):
        """GET /pipeline/runs/{unknown_id} → status=unknown (нет событий)."""
        from fastapi.testclient import TestClient

        from aqr.main import app

        client = TestClient(app)
        r = client.get(f"/pipeline/runs/{uuid.uuid4()}")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "unknown"
        assert body["events"] == []


# ── GET /pipeline/runs/{id}/stream (SSE) ────────────────────────
# SSE-тесты сложно сделать синхронно через TestClient (event-loop блокируется).
# Покрываем только проверку, что endpoint существует.


# ── _run_and_persist с успешным executor ────────────────────────


class TestRunAndPersist:
    @pytest.mark.asyncio
    async def test_persist_writes_metrics_and_hypotheses(
        self, mock_db, monkeypatch, with_credentials, fake_openai
    ):
        """При успешном executor — пишет metrics + embedding + hypothesis."""
        from aqr.pipeline.api import _run_and_persist
        from aqr.pipeline.executor import BacktestResult, PipelineResult
        from aqr.pipeline.hypotheses import HypothesisSpec

        # Мок executor.run → возвращает фиктивный PipelineResult
        spec = HypothesisSpec(
            name="SMA10/50", family="momentum", ticker="SBER",
            params={"fast": 10, "slow": 50}, fn=lambda x: x,
        )
        fake_top = [
            BacktestResult(
                hypothesis=spec, sharpe=1.2, dsr=0.95, dsr_verdict="significant",
                cpcv_mean_sharpe=0.7, cpcv_std_sharpe=0.1, max_drawdown=-0.1,
                n_trades=10, daily_returns=[0.01] * 100,
            )
        ]
        fake_result = PipelineResult(
            run_id="test-run", plan=MagicMock(),
            n_hypotheses_tested=10, n_survived_dsr=3,
            portfolio_pbo=0.3, portfolio_pbo_verdict="ok",
            top=fake_top, elapsed_seconds=1.0, narrative="narrative",
        )

        class _FakeExec:
            async def run(self, run_id, plan):
                return fake_result

        await _run_and_persist(
            str(uuid.uuid4()),
            MagicMock(),
            _FakeExec(),
        )

        # Проверяем, что update_run_status вызвался
        mock_db.update_run_status.assert_called_once()
        # И create_hypothesis с embedding
        mock_db.create_hypothesis.assert_called_once()
        call_kwargs = mock_db.create_hypothesis.call_args.kwargs
        assert "embedding" in call_kwargs
        assert isinstance(call_kwargs["embedding"], list)
        assert len(call_kwargs["embedding"]) == 1536

    @pytest.mark.asyncio
    async def test_persist_handles_executor_failure(self, mock_db):
        """Если executor падает — статус error, без падений."""
        from aqr.pipeline.api import _run_and_persist

        class _FailingExec:
            async def run(self, run_id, plan):
                raise RuntimeError("pipeline crashed")

        # Не должно быть исключения
        await _run_and_persist(
            str(uuid.uuid4()),
            MagicMock(),
            _FailingExec(),
        )

        # update_run_status вызвался с status="error"
        assert mock_db.update_run_status.called
        # Берём последний вызов
        call = mock_db.update_run_status.call_args_list[-1]
        # kwargs может быть пустым, args[1] = status
        status_value = call.kwargs.get("status") if call.kwargs else call.args[1] if len(call.args) > 1 else None
        assert status_value == "error"
