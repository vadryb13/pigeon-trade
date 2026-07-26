"""Tests for InsightReviewer — LLM-обзор поверх детерминистичных инсайтов."""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

from aqr.pipeline import InsightReviewer
from aqr.pipeline.executor import BacktestResult, PipelineResult
from aqr.pipeline.hypotheses import HypothesisSpec
from aqr.pipeline.planner import ResearchPlan


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


def _install_fake_litellm(monkeypatch, response_content: str) -> MagicMock:
    """Ставит в sys.modules фейковый litellm с litellm.completion.

    Работает и когда настоящий litellm не установлен (базовое окружение),
    и когда установлен (мок замещает completion).
    """
    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock()]
    fake_resp.choices[0].message.content = response_content

    fake_module = types.ModuleType("litellm")
    fake_module.completion = MagicMock(return_value=fake_resp)
    monkeypatch.setitem(sys.modules, "litellm", fake_module)
    return fake_module.completion


def test_reviewer_returns_empty_without_key(monkeypatch):
    """Без ключа reviewer тихо возвращает [] — pipeline не ломается."""
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GIGACHAT_CREDENTIALS"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("AQR_LLM_MODEL", "claude-3-5-sonnet-20241022")

    reviewer = InsightReviewer()
    result = reviewer.review(_fake_result(), ["Тестовый инсайт"])
    assert result == []


def test_reviewer_returns_empty_without_model(monkeypatch):
    """Без модели reviewer возвращает []."""
    monkeypatch.delenv("AQR_LLM_MODEL", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    reviewer = InsightReviewer()
    result = reviewer.review(_fake_result(), ["Тестовый инсайт"])
    assert result == []


def test_reviewer_returns_empty_when_no_top(monkeypatch):
    """При пустом top LLM не вызывается — экономим токены."""
    monkeypatch.setenv("AQR_LLM_MODEL", "test-model")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    mock_completion = _install_fake_litellm(monkeypatch, '{"observations": ["nope"]}')

    r = _fake_result()
    r.top = []
    reviewer = InsightReviewer()
    result = reviewer.review(r, [])
    assert result == []
    mock_completion.assert_not_called()


def test_reviewer_parses_llm_response(monkeypatch):
    """С мок-LLM возвращает observations."""
    monkeypatch.setenv("AQR_LLM_MODEL", "test-model")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    mock_completion = _install_fake_litellm(
        monkeypatch,
        '{"observations": ['
        '"Топ забит одним тикером SBER — edge не диверсифицирован.",'
        '"Только 100 баров — маловато для доверия к DSR."'
        ']}',
    )

    reviewer = InsightReviewer()
    result = reviewer.review(_fake_result(), ["Ранее найденный инсайт"])

    assert len(result) == 2
    assert "SBER" in result[0]
    mock_completion.assert_called_once()

    # Убедимся, что payload содержит нужные поля
    call = mock_completion.call_args
    user_msg = call.kwargs["messages"][1]["content"]
    assert "SBER" in user_msg
    assert "deterministic_insights" in user_msg


def test_reviewer_caps_at_3(monkeypatch):
    """Если LLM вернул 5+ наблюдений — берём максимум 3."""
    monkeypatch.setenv("AQR_LLM_MODEL", "test-model")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    _install_fake_litellm(
        monkeypatch,
        '{"observations": ["a", "b", "c", "d", "e"]}',
    )

    reviewer = InsightReviewer()
    result = reviewer.review(_fake_result(), [])
    assert len(result) == 3


def test_reviewer_survives_bad_llm_response(monkeypatch):
    """Кривой JSON от LLM → пустой список, не exception."""
    monkeypatch.setenv("AQR_LLM_MODEL", "test-model")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    _install_fake_litellm(monkeypatch, "не JSON, а текст")

    reviewer = InsightReviewer()
    result = reviewer.review(_fake_result(), [])
    assert result == []


def test_reviewer_trims_long_observations(monkeypatch):
    """Слишком длинное наблюдение обрезается до 400 символов."""
    monkeypatch.setenv("AQR_LLM_MODEL", "test-model")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    huge = "x" * 1000
    _install_fake_litellm(
        monkeypatch,
        '{"observations": ["' + huge + '"]}',
    )

    reviewer = InsightReviewer()
    result = reviewer.review(_fake_result(), [])
    assert len(result) == 1
    assert len(result[0]) <= 400
