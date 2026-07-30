"""End-to-end тест сквозного пайплайна со строгим режимом.

Без fallback. Все LLM-вызовы — через мок litellm.
"""
from __future__ import annotations

import pytest

from aqr.pipeline import ResearchPlanner
from aqr.pipeline.executor import BacktestResult, PipelineResult
from aqr.pipeline.hypotheses import HypothesisSpec
from aqr.pipeline.planner import ResearchPlan


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
        self, with_credentials, fake_litellm, monkeypatch
    ):
        """С credentials + мок LLM — план парсится из JSON-ответа."""
        fake_litellm(
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
        self, with_credentials, fake_litellm, monkeypatch
    ):
        """LLM вернул неполный JSON — дефолты применяются."""
        fake_litellm("{}")
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
