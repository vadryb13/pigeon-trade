"""Tests for InsightReviewer — LLM-обзор поверх детерминистичных инсайтов.

Строгий режим: всегда зовёт LLM с per-session credentials.
Без credentials или пустом top → raise.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from aqr.pipeline import InsightReviewer
from aqr.pipeline.executor import BacktestResult, PipelineResult
from aqr.pipeline.hypotheses import HypothesisSpec
from aqr.pipeline.planner import ResearchPlan


@pytest.fixture(autouse=True)
def _stable_secret(monkeypatch):
    monkeypatch.setenv("AQR_SESSION_SECRET", "test-secret-padded-to-32-bytes-base64==")


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


def _install_fake_litellm(monkeypatch, response_content: str) -> AsyncMock:
    """Ставит в sys.modules фейковый litellm с litellm.acompletion (async)."""
    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock()]
    fake_resp.choices[0].message.content = response_content

    fake_module = types.ModuleType("litellm")
    fake_module.acompletion = AsyncMock(return_value=fake_resp)
    monkeypatch.setitem(sys.modules, "litellm", fake_module)
    return fake_module.acompletion


async def test_reviewer_raises_without_credentials(monkeypatch):
    """Без active credentials в ContextVar → RuntimeError."""
    from aqr.graph.context import current_credentials

    assert current_credentials() is None

    reviewer = InsightReviewer()
    with pytest.raises(RuntimeError, match="credentials not configured"):
        await reviewer.review(_fake_result(), ["test"])


async def test_reviewer_raises_when_top_is_empty(monkeypatch, with_credentials):
    """Пустой result.top → ValueError (экономим токены до LLM-вызова)."""
    monkeypatch.setenv("AQR_LLM_MODEL", "test-model")
    mock_completion = _install_fake_litellm(monkeypatch, '{"observations": []}')

    r = _fake_result()
    r.top = []
    reviewer = InsightReviewer()
    with pytest.raises(ValueError, match="result.top is empty"):
        await reviewer.review(r, [])
    mock_completion.assert_not_called()


async def test_reviewer_parses_llm_response(monkeypatch, with_credentials):
    """С credentials и мок-LLM возвращает observations."""
    monkeypatch.setenv("AQR_LLM_MODEL", "test-model")
    mock_completion = _install_fake_litellm(
        monkeypatch,
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


async def test_reviewer_caps_at_3(monkeypatch, with_credentials):
    """Если LLM вернул 5+ наблюдений — берём максимум 3."""
    monkeypatch.setenv("AQR_LLM_MODEL", "test-model")
    _install_fake_litellm(
        monkeypatch,
        '{"observations": ["a", "b", "c", "d", "e"]}',
    )

    reviewer = InsightReviewer()
    result = await reviewer.review(_fake_result(), [])
    assert len(result) == 3


async def test_reviewer_propagates_bad_json(monkeypatch, with_credentials):
    """Кривой JSON от LLM → ValueError (теперь не глотается)."""
    monkeypatch.setenv("AQR_LLM_MODEL", "test-model")
    _install_fake_litellm(monkeypatch, "не JSON, а текст")

    reviewer = InsightReviewer()
    with pytest.raises(ValueError, match="Expecting value"):
        await reviewer.review(_fake_result(), [])


async def test_reviewer_trims_long_observations(monkeypatch, with_credentials):
    """Слишком длинное наблюдение обрезается до 400 символов."""
    monkeypatch.setenv("AQR_LLM_MODEL", "test-model")
    huge = "x" * 1000
    _install_fake_litellm(
        monkeypatch,
        '{"observations": ["' + huge + '"]}',
    )

    reviewer = InsightReviewer()
    result = await reviewer.review(_fake_result(), [])
    assert len(result) == 1
    assert len(result[0]) <= 400
