"""Tests for v0.4 API endpoints."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from aqr.graph.context import reset_credentials, set_credentials
from aqr.registry import DecryptedSettings


@pytest.fixture(autouse=True)
def _stable_secret(monkeypatch):
    monkeypatch.setenv("AQR_SESSION_SECRET", "test-secret-padded-to-32-bytes-base64==")


@pytest.fixture(autouse=True)
def _clear_registry(monkeypatch):
    from aqr.tools import reset_for_testing
    reset_for_testing()


@pytest.fixture
def with_credentials():
    creds = DecryptedSettings(
        session_id="test-api",
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
def app(monkeypatch):
    """Import the FastAPI app with startup validation mocked."""
    monkeypatch.setattr("aqr.startup.validate_runtime", AsyncMock())
    from aqr.main import app
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


# ── POST /team/run ───────────────────────────────────────────────

class TestTeamRun:
    @pytest.mark.asyncio
    async def test_post_team_run_returns_summary(self, with_credentials, client, monkeypatch):
        """POST /team/run returns TeamResult data."""
        from aqr.agents.orchestrator import TeamResult

        async def fake_run_team(**kw):
            return TeamResult(
                ok=True,
                goal=kw.get("goal", ""),
                plan={"tickers": ["SBER"]},
                context={},
                results=[{"dsr": 1.2}],
                validation={"pbo": 0.3, "n_tested": 1, "n_survived": 1,
                            "survival_rate": 1.0, "aggregate": {}, "recommendations": []},
                narrative="Тестовый отчёт.",
                insights=["Инсайт 1"],
                summary="Лучшая: momentum/SBER DSR=1.20 | Проверено: 1 | Выжило: 1 | PBO=0.30",
                top_results=[{"family": "momentum", "ticker": "SBER", "dsr": 1.2}],
                elapsed_seconds=0.5,
                n_tested=1,
                n_survived=1,
                error="",
                agent_errors=[],
            )

        monkeypatch.setattr(
            "aqr.agents.orchestrator.run_team",
            fake_run_team,
        )

        resp = client.post(
            "/team/run",
            json={
                "goal": "проверь momentum на Сбере",
                "session_id": "test-api",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "Тестовый отчёт" in data["narrative"]
        assert data["summary"]

    @pytest.mark.asyncio
    async def test_post_team_run_with_optional_fields(self, with_credentials, client, monkeypatch):
        """Team run accepts optional tickers/families."""
        from aqr.agents.orchestrator import TeamResult

        async def fake_run_team(**kw):
            return TeamResult(
                ok=True,
                goal=kw.get("goal", ""),
                plan={"tickers": kw.get("tickers", ["SBER"])},
                context={}, results=[],
                validation={"pbo": 0.0, "n_tested": 0, "n_survived": 0,
                            "survival_rate": 0.0, "aggregate": {}, "recommendations": []},
                narrative="", insights=[], summary="",
                top_results=[], elapsed_seconds=0.0,
                n_tested=0, n_survived=0, error="", agent_errors=[],
            )

        monkeypatch.setattr(
            "aqr.agents.orchestrator.run_team",
            fake_run_team,
        )

        resp = client.post(
            "/team/run",
            json={
                "goal": "тест",
                "tickers": ["GAZP"],
                "families": ["mean_reversion"],
            },
        )
        assert resp.status_code == 200


# ── POST /executor/nautilus ──────────────────────────────────────

class TestExecutorNautilus:
    @pytest.mark.asyncio
    async def test_post_executor_nautilus(self, with_credentials, client, monkeypatch):
        """POST /executor/nautilus returns BacktestResult dict."""
        from aqr.types import BacktestResult

        async def fake_execute(**kw):
            from aqr.pipeline.hypotheses import HypothesisSpec
            return BacktestResult(
                hypothesis=HypothesisSpec(
                    name="test", family="momentum", ticker="SBER",
                    params={"fast": 10}, fn=lambda x: x,
                ),
                sharpe=1.5, dsr=1.2, dsr_verdict="significant",
                cpcv_mean_sharpe=0.9, cpcv_std_sharpe=0.3,
                max_drawdown=-0.15, n_trades=42,
                daily_returns=[0.001] * 100,
            )

        monkeypatch.setattr(
            "aqr.executor.nautilus.execute_with_slippage",
            fake_execute,
        )

        resp = client.post(
            "/executor/nautilus",
            json={
                "hypothesis": {"family": "momentum", "ticker": "SBER", "params": {"fast": 10}},
                "prices": [100.0] * 500,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["sharpe"] == 1.5
        assert data["dsr"] == 1.2
        assert data["dsr_verdict"] == "significant"


# ── POST /mcp/rpc ────────────────────────────────────────────────

class TestMCPRpc:
    @pytest.mark.asyncio
    async def test_post_mcp_rpc(self, client, monkeypatch):
        """POST /mcp/rpc returns JSON-RPC response."""
        async def fake_dispatch(method, params):
            return {"jsonrpc": "2.0", "result": "ok", "id": 1}

        monkeypatch.setattr(
            "aqr.mcp.server.dispatch",
            fake_dispatch,
        )

        resp = client.post(
            "/mcp/rpc",
            json={"method": "ping", "params": {}},
        )
        assert resp.status_code == 200
        assert resp.json()["result"] == "ok"
