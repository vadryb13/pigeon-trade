"""Tests for InsightReviewer — LLM-обзор поверх детерминистичных инсайтов.

Строгий режим: всегда зовёт LLM с per-session credentials.
Без credentials или пустом top → raise.
"""
from __future__ import annotations

import pytest

from aqr.pipeline import InsightReviewer
from aqr.pipeline.executor import BacktestResult, PipelineResult
from aqr.pipeline.hypotheses import HypothesisSpec
from aqr.pipeline.planner import ResearchPlan


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


def _fake_result() -> PipelineResult:
    plan = ResearchPlan(
        goal="проверь momentum на Сбере",
        tickers=["SBER"],
        hypothesis_families=["momentum"],
    )

    def _dummy_signal(_prices):
        raise NotImplementedError

    spec = HypothesisSpec(
        name="SMA5/50",
        family="momentum",
        ticker="SBER",
        params={"fast": 5, "slow": 50},
        fn=_dummy_signal,
    )
    top = [
        BacktestResult(
            hypothesis=spec,
            sharpe=1.5,
            dsr=0.4,
            dsr_verdict="not_significant",
            cpcv_mean_sharpe=1.2,
            cpcv_std_sharpe=0.3,
            max_drawdown=-0.15,
            n_trades=42,
            daily_returns=[0.001] * 100,
        )
    ]
    return PipelineResult(
        run_id="test-run",
        plan=plan,
        n_hypotheses_tested=20,
        n_survived_dsr=0,
        portfolio_pbo=0.35,
        portfolio_pbo_verdict="suspicious",
        top=top,
        elapsed_seconds=5.0,
    )


async def test_reviewer_raises_without_credentials(monkeypatch):
    """Без active credentials в ContextVar → RuntimeError."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from aqr.graph.context import current_credentials

    assert current_credentials() is None

    reviewer = InsightReviewer()
    with pytest.raises(RuntimeError, match="no credentials available"):
        await reviewer.review(_fake_result(), ["test"])


async def test_reviewer_raises_when_top_is_empty(monkeypatch, with_credentials, fake_litellm):
    """Пустой result.top → ValueError (экономим токены до LLM-вызова)."""
    monkeypatch.setenv("AQR_LLM_MODEL", "test-model")
    mock_completion = fake_litellm('{"observations": []}')

    r = _fake_result()
    r.top = []
    reviewer = InsightReviewer()
    with pytest.raises(ValueError, match="result.top is empty"):
        await reviewer.review(r, [])
    mock_completion.assert_not_called()


async def test_reviewer_parses_llm_response(monkeypatch, with_credentials, fake_litellm):
    """С credentials и мок-LLM возвращает observations."""
    monkeypatch.setenv("AQR_LLM_MODEL", "test-model")
    mock_completion = fake_litellm(
        '{"observations": ['
        '"Топ забит одним тикером SBER — edge не диверсифицирован.",'
        '"Только 100 баров — маловато для доверия к DSR."'
        ']}',
    )

    reviewer = InsightReviewer()
    result = await reviewer.review(_fake_result(), ["Ранее найденный инсайт"])

    assert len(result) == 2
    assert "SBER" in result[0]
    mock_completion.assert_awaited_once()

    call = mock_completion.call_args
    user_msg = call.kwargs["messages"][1]["content"]
    assert "SBER" in user_msg
    assert "deterministic_insights" in user_msg


async def test_reviewer_caps_at_3(monkeypatch, with_credentials, fake_litellm):
    """Если LLM вернул 5+ наблюдений — берём максимум 3."""
    monkeypatch.setenv("AQR_LLM_MODEL", "test-model")
    fake_litellm(
        '{"observations": ["a", "b", "c", "d", "e"]}',
    )

    reviewer = InsightReviewer()
    result = await reviewer.review(_fake_result(), [])
    assert len(result) == 3


async def test_reviewer_propagates_bad_json(monkeypatch, with_credentials, fake_litellm):
    """Кривой JSON от LLM → ValueError (теперь не глотается)."""
    monkeypatch.setenv("AQR_LLM_MODEL", "test-model")
    fake_litellm("не JSON, а текст")

    reviewer = InsightReviewer()
    with pytest.raises(ValueError, match="Expecting value"):
        await reviewer.review(_fake_result(), [])


async def test_reviewer_trims_long_observations(monkeypatch, with_credentials, fake_litellm):
    """Слишком длинное наблюдение обрезается до 400 символов."""
    monkeypatch.setenv("AQR_LLM_MODEL", "test-model")
    huge = "x" * 1000
    fake_litellm(
        '{"observations": ["' + huge + '"]}',
    )

    reviewer = InsightReviewer()
    result = await reviewer.review(_fake_result(), [])
    assert len(result) == 1
    assert len(result[0]) <= 400
