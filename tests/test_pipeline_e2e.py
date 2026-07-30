"""End-to-end тест сквозного пайплайна со строгим режимом.

Без fallback. Все LLM-вызовы — через мок litellm.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from aqr.pipeline import ResearchPlanner
from aqr.pipeline.executor import BacktestResult, PipelineResult
from aqr.pipeline.hypotheses import HypothesisSpec
from aqr.pipeline.planner import ResearchPlan


@pytest.fixture(autouse=True)
def _stable_secret(monkeypatch):
    monkeypatch.setenv("AQR_SESSION_SECRET", "test-secret-padded-to-32-bytes-base64==")


def _fake_credentials():
    from aqr.registry import DecryptedSettings
    return DecryptedSettings(
        session_id="alice",
        llm_model="claude-3-5-sonnet-20241022",
        llm_api_key="sk-ant-fake",
        openai_api_key="sk-oai-fake",
        invest_token="t.INVEST_TOKEN_fake",
        invest_sandbox=True,
    )


@pytest.fixture
def active_credentials(monkeypatch):
    """Устанавливает credentials в ContextVar, очищает на teardown."""
    from aqr.graph.context import reset_credentials, set_credentials

    token = set_credentials(_fake_credentials())
    yield _fake_credentials()
    reset_credentials(token)


@pytest.fixture
def fake_litellm(monkeypatch):
    """Подменяет litellm.acompletion фейковым async-моком."""
    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock()]
    fake_resp.choices[0].message.content = "{}"
    fake_module = types.ModuleType("litellm")
    fake_module.acompletion = AsyncMock(return_value=fake_resp)
    monkeypatch.setitem(sys.modules, "litellm", fake_module)
    return fake_module.acompletion


class TestPlannerRequiresCredentials:
    async def test_raises_without_credentials(self, monkeypatch):
        """Без active credentials → RuntimeError."""
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from aqr.graph.context import current_credentials

        assert current_credentials() is None

        planner = ResearchPlanner()
        with pytest.raises(RuntimeError, match="no credentials available"):
            await planner.plan("проверь momentum на Сбере")


class TestNarratorRequiresCredentials:
    async def test_raises_without_credentials(self, monkeypatch):
        """Narrator без credentials → raise."""
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from aqr.graph.context import current_credentials

        assert current_credentials() is None

        plan = ResearchPlan(
            goal="тест",
            tickers=["SBER"],
            hypothesis_families=["momentum"],
        )

        def _dummy_signal(_):
            raise NotImplementedError

        spec = HypothesisSpec(
            name="SMA5/50", family="momentum", ticker="SBER",
            params={"fast": 5}, fn=_dummy_signal,
        )
        result = PipelineResult(
            run_id="test", plan=plan, n_hypotheses_tested=1, n_survived_dsr=0,
            portfolio_pbo=0.0, portfolio_pbo_verdict="robust",
            top=[BacktestResult(
                hypothesis=spec, sharpe=1.0, dsr=0.5,
                dsr_verdict="significant", cpcv_mean_sharpe=0.8,
                cpcv_std_sharpe=0.2, max_drawdown=-0.1, n_trades=10,
                daily_returns=[0.001] * 50,
            )],
        )

        from aqr.pipeline import Narrator

        narrator = Narrator()
        with pytest.raises(RuntimeError, match="no credentials available"):
            await narrator.narrate(result)


class TestPlannerWithMockedLLM:
    async def test_parses_llm_json_response(
        self, active_credentials, fake_litellm, monkeypatch
    ):
        """С credentials + мок LLM — план парсится из JSON-ответа."""
        fake_litellm.return_value.choices[0].message.content = (
            '{"tickers": ["GAZP"], "timeframe": "H1", '
            '"start_date": "2024-01-01", "end_date": "2024-12-31", '
            '"hypothesis_families": ["breakout"], '
            '"n_hypotheses": 30, "rationale": "тестовый план"}'
        )
        monkeypatch.setenv("AQR_LLM_MODEL", "claude-3-5-sonnet-20241022")

        plan = await ResearchPlanner().plan("тест")
        assert plan.tickers == ["GAZP"]
        assert plan.timeframe == "H1"
        assert plan.hypothesis_families == ["breakout"]
        assert plan.n_hypotheses == 30

    async def test_uses_defaults_when_json_incomplete(
        self, active_credentials, fake_litellm, monkeypatch
    ):
        """LLM вернул неполный JSON — дефолты применяются."""
        fake_litellm.return_value.choices[0].message.content = "{}"
        monkeypatch.setenv("AQR_LLM_MODEL", "claude-3-5-sonnet-20241022")

        plan = await ResearchPlanner().plan("тест")
        # Дефолты
        assert plan.tickers == ["SBER", "GAZP", "LKOH"]
        assert plan.timeframe == "D1"
        assert plan.hypothesis_families == ["momentum", "mean_reversion"]
        assert plan.n_hypotheses == 20


class TestPipelineRequiresCredentials:
    async def test_pipeline_sends_settings_error(self, monkeypatch):
        """Без credentials — planner и narrator падают (проверяется в отдельных классах)."""
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from aqr.graph.context import current_credentials

        assert current_credentials() is None

        planner = ResearchPlanner()
        with pytest.raises(RuntimeError, match="no credentials available"):
            await planner.plan("тест")
