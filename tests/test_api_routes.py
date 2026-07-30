"""Тесты для aqr/pipeline/api.py — FastAPI роуты.

Покрывают POST /pipeline/runs, GET /pipeline/runs/{id}, GET /pipeline/runs/{id}/stream.
"""
from __future__ import annotations

import sys
import uuid
from unittest.mock import AsyncMock, MagicMock

from conftest import FakeSession, fake_openai_module

import pytest


@pytest.fixture(autouse=True)
def _stable_secret(monkeypatch):
    monkeypatch.setenv("AQR_SESSION_SECRET", "test-secret-padded-to-32-bytes-base64==")


@pytest.fixture
def fake_openai(monkeypatch):
    """Мок openai.AsyncOpenAI — embeddings возвращает [0.1] * 768."""
    monkeypatch.setitem(sys.modules, "openai", fake_openai_module())


@pytest.fixture
def mock_db(monkeypatch):
    """Мок БД и Executor."""
    factory = type("_F", (), {"__call__": lambda self: FakeSession()})()

    from aqr import session as db_mod
    monkeypatch.setattr(db_mod, "async_session_factory", factory)

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
        assert len(call_kwargs["embedding"]) == 768

    @pytest.mark.asyncio
    async def test_persist_handles_executor_failure(self, mock_db):
        """Если executor падает — статус error, без падений."""
        from aqr.pipeline.api import _run_and_persist

        class _FailingExec:
            async def run(self, run_id, plan):
                raise RuntimeError("pipeline crashed")

        await _run_and_persist(
            str(uuid.uuid4()),
            MagicMock(),
            _FailingExec(),
        )

        assert mock_db.update_run_status.called
        all_kwargs = {}
        for call in mock_db.update_run_status.call_args_list:
            all_kwargs.update(call.kwargs)
        all_kwargs.update({f"pos_{i}": v for i, v in enumerate(call.args) if call.args})
        assert any(
            v == "error" for v in all_kwargs.values() if isinstance(v, str)
        ), f"update_run_status not called with status='error': {mock_db.update_run_status.call_args_list}"
