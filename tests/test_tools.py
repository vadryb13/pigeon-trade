"""Изолированные тесты на инструменты ToolRegistry."""
from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from aqr.tools import ToolSpec, registry, reset_for_testing
from aqr.tools.register import register_all

# ── Фикстуры ────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _stable_secret(monkeypatch):
    monkeypatch.setenv("AQR_SESSION_SECRET", "test-secret-padded-to-32-bytes-base64==")


@pytest.fixture(autouse=True)
def _clean_registry():
    """Очищаем реестр перед каждым тестом и перерегистрируем."""
    reset_for_testing()
    register_all()
    yield
    reset_for_testing()
    register_all()


@pytest.fixture
def with_credentials():
    """Устанавливает credentials в ContextVar, очищает на teardown."""
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
def fake_litellm(monkeypatch):
    """Мок litellm.acompletion с настраиваемым JSON-ответом."""
    def _install(content: str):
        fake_resp = MagicMock()
        fake_resp.choices = [MagicMock()]
        fake_resp.choices[0].message.content = content

        fake_module = types.ModuleType("litellm")
        fake_module.acompletion = AsyncMock(return_value=fake_resp)
        monkeypatch.setitem(sys.modules, "litellm", fake_module)
        return fake_module.acompletion

    return _install


@pytest.fixture
def synthetic_prices():
    """Генерирует синтетический ряд цен (500 баров GBM)."""
    rng = np.random.default_rng(42)
    n = 500
    ret = rng.normal(0.0005, 0.015, n)
    ret[100:200] += 0.002
    px = 100 * np.exp(np.cumsum(ret))
    return px.tolist()


@pytest.fixture
def sample_hypothesis():
    return {
        "name": "SMA10/50",
        "family": "momentum",
        "ticker": "SBER",
        "params": {"fast": 10, "slow": 50},
    }


# ── ToolRegistry ────────────────────────────────────────────────

class TestToolRegistry:
    def test_list_all_returns_13_tools(self):
        """После register_all() в реестре 13 инструментов."""
        tools = registry.list_all()
        assert len(tools) == 13, f"Expected 13 tools, got {len(tools)}"

    def test_list_all_returns_tool_specs(self):
        """list_all() возвращает список ToolSpec."""
        tools = registry.list_all()
        for t in tools:
            assert isinstance(t, ToolSpec)

    def test_get_returns_tool(self):
        """get() по имени возвращает ToolSpec."""
        tool = registry.get("plan_research")
        assert tool is not None
        assert tool.name == "plan_research"
        assert tool.category == "pipeline"

    def test_get_nonexistent_returns_none(self):
        """get() несуществующего инструмента возвращает None."""
        assert registry.get("nonexistent_tool") is None

    def test_list_for_llm_has_keys(self):
        """list_for_llm() возвращает список словарей с name и description."""
        llm_list = registry.list_for_llm()
        assert len(llm_list) == 13
        for entry in llm_list:
            assert "name" in entry
            assert "description" in entry
            assert "parameters" in entry

    def test_tool_names_are_unique(self):
        """Имена инструментов уникальны."""
        names = [t.name for t in registry.list_all()]
        assert len(names) == len(set(names))

    def test_all_expected_tools_registered(self):
        """Все 13 ожидаемых имён присутствуют."""
        expected = {
            "plan_research", "load_prices", "generate_hypotheses",
            "backtest_one", "validate_portfolio", "extract_insights",
            "review_insights", "narrate",
            "get_run", "compare_runs", "list_runs",
            "search_similar_hypotheses", "find_duplicates",
        }
        actual = {t.name for t in registry.list_all()}
        assert actual == expected

    def test_pipeline_tools_count(self):
        """8 инструментов категории pipeline, 5 — storage."""
        pipeline = [t for t in registry.list_all() if t.category == "pipeline"]
        storage = [t for t in registry.list_all() if t.category == "storage"]
        assert len(pipeline) == 8
        assert len(storage) == 5


# ── plan_research ───────────────────────────────────────────────

class TestPlanResearch:
    @pytest.mark.asyncio
    async def test_plan_momentum_sber(self, monkeypatch, with_credentials, fake_litellm):
        """План для 'проверь momentum на Сбере' с моком LLM и мок БД (пустой dedup)."""
        from aqr import db as db_mod
        from aqr.registry import RegistryStore

        # Мок DB: возвращает пустой результат search_similar (нет похожих)
        class _EmptySession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return None

            async def execute(self, *a, **kw):
                class _R:
                    def all(self):
                        return []
                return _R()

        class _EmptyFactory:
            def __call__(self):
                return _EmptySession()

        monkeypatch.setattr(db_mod, "_async_session_factory", _EmptyFactory())

        # Мок openai (модуль может не быть установлен)
        import types
        from unittest.mock import MagicMock

        class _FakeEmbeddingsAPI:
            async def create(self, *, model, input):
                return MagicMock(data=[MagicMock(embedding=[0.1] * 1536)])

        class _FakeAsyncOpenAI:
            def __init__(self, **kw):
                self.embeddings = _FakeEmbeddingsAPI()

        fake_openai = types.ModuleType("openai")
        fake_openai.AsyncOpenAI = _FakeAsyncOpenAI
        monkeypatch.setitem(sys.modules, "openai", fake_openai)

        fake_litellm(
            '{"tickers": ["SBER"], "hypothesis_families": ["momentum"], '
            '"n_hypotheses": 20, "rationale": "test"}'
        )
        monkeypatch.setenv("AQR_LLM_MODEL", "claude-3-5-sonnet-20241022")

        tool = registry.get("plan_research")
        result = await tool.fn(goal="проверь momentum на Сбере")
        assert "tickers" in result
        assert "hypothesis_families" in result
        assert "n_hypotheses" in result
        assert "momentum" in result["hypothesis_families"]
        # Пустой результат search_similar — нет dedup_warning
        assert "dedup_warning" not in result

    @pytest.mark.asyncio
    async def test_plan_blue_chips(
        self, monkeypatch, with_credentials, fake_litellm
    ):
        from aqr import db as db_mod

        class _EmptySession:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return None
            async def execute(self, *a, **kw):
                class _R:
                    def all(self):
                        return []
                return _R()

        class _EmptyFactory:
            def __call__(self):
                return _EmptySession()

        monkeypatch.setattr(db_mod, "_async_session_factory", _EmptyFactory())

        import types
        from unittest.mock import MagicMock

        class _FakeEmbeddingsAPI:
            async def create(self, *, model, input):
                return MagicMock(data=[MagicMock(embedding=[0.1] * 1536)])

        class _FakeAsyncOpenAI:
            def __init__(self, **kw):
                self.embeddings = _FakeEmbeddingsAPI()

        fake_openai = types.ModuleType("openai")
        fake_openai.AsyncOpenAI = _FakeAsyncOpenAI
        monkeypatch.setitem(sys.modules, "openai", fake_openai)

        fake_litellm(
            '{"tickers": ["SBER", "GAZP", "LKOH"], '
            '"hypothesis_families": ["mean_reversion"], "n_hypotheses": 20}'
        )
        monkeypatch.setenv("AQR_LLM_MODEL", "claude-3-5-sonnet-20241022")

        tool = registry.get("plan_research")
        result = await tool.fn(goal="проверь mean reversion на голубых фишках")
        assert len(result["tickers"]) > 1, "Голубые фишки — несколько тикеров"
        assert "mean_reversion" in result["hypothesis_families"]

    @pytest.mark.asyncio
    async def test_plan_returns_dict(self, monkeypatch, with_credentials, fake_litellm):
        from aqr import db as db_mod

        class _EmptySession:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return None
            async def execute(self, *a, **kw):
                class _R:
                    def all(self):
                        return []
                return _R()

        class _EmptyFactory:
            def __call__(self):
                return _EmptySession()

        monkeypatch.setattr(db_mod, "_async_session_factory", _EmptyFactory())

        import types
        from unittest.mock import MagicMock

        class _FakeEmbeddingsAPI:
            async def create(self, *, model, input):
                return MagicMock(data=[MagicMock(embedding=[0.1] * 1536)])

        class _FakeAsyncOpenAI:
            def __init__(self, **kw):
                self.embeddings = _FakeEmbeddingsAPI()

        fake_openai = types.ModuleType("openai")
        fake_openai.AsyncOpenAI = _FakeAsyncOpenAI
        monkeypatch.setitem(sys.modules, "openai", fake_openai)

        fake_litellm('{"tickers": ["GAZP"], "hypothesis_families": ["breakout"]}')
        monkeypatch.setenv("AQR_LLM_MODEL", "claude-3-5-sonnet-20241022")

        tool = registry.get("plan_research")
        result = await tool.fn(goal="протестируй breakout на Газпроме")
        assert isinstance(result, dict)
        assert result.get("start_date")
        assert result.get("end_date")


# ── load_prices ─────────────────────────────────────────────────

class TestLoadPrices:
    @pytest.mark.asyncio
    async def test_load_with_tinvest(self, monkeypatch, with_credentials):
        """load_prices через T-Invest (мок). Без fallback."""
        import uuid

        # Уникальный кэш на каждый тест
        cache_path = f"/tmp/aqr_test_load_{uuid.uuid4().hex[:8]}.duckdb"
        monkeypatch.setenv("AQR_CACHE_DIR", cache_path)

        # Мок TInvestAdapter
        import pandas as pd

        from aqr.data import tinvest as tinvest_mod

        class _FakeAdapter:
            def __init__(self, *a, **kw):
                pass

            def candles(self, ticker, from_date, to_date, interval="D1"):
                rng = pd.date_range("2023-01-02", periods=500, freq="B")
                px = [100 + i * 0.1 for i in range(500)]
                return pd.DataFrame({
                    "open": px, "high": px, "low": px,
                    "close": px, "volume": [0] * 500,
                }, index=rng)

        monkeypatch.setattr(tinvest_mod, "TInvestAdapter", _FakeAdapter)

        tool = registry.get("load_prices")
        result = await tool.fn(tickers=["SBER", "GAZP"])

        assert isinstance(result, dict)
        assert "SBER" in result
        assert "GAZP" in result
        for px in result.values():
            assert isinstance(px, list)
            assert len(px) == 500
            assert all(isinstance(p, (int, float)) for p in px)

    @pytest.mark.asyncio
    async def test_load_single_ticker(self, monkeypatch, with_credentials):
        """Один тикер — T-Invest возвращает >=100 баров."""
        import uuid

        cache_path = f"/tmp/aqr_test_single_{uuid.uuid4().hex[:8]}.duckdb"
        monkeypatch.setenv("AQR_CACHE_DIR", cache_path)

        import pandas as pd

        from aqr.data import tinvest as tinvest_mod

        class _FakeAdapter:
            def __init__(self, *a, **kw):
                pass

            def candles(self, ticker, from_date, to_date, interval="D1"):
                rng = pd.date_range("2023-01-02", periods=500, freq="B")
                px = [100.0] * 500
                return pd.DataFrame({
                    "open": px, "high": px, "low": px,
                    "close": px, "volume": [0] * 500,
                }, index=rng)

        monkeypatch.setattr(tinvest_mod, "TInvestAdapter", _FakeAdapter)

        tool = registry.get("load_prices")
        result = await tool.fn(tickers=["LKOH"])
        assert len(result) == 1
        assert "LKOH" in result
        assert len(result["LKOH"]) == 500


# ── PIT safety net ──────────────────────────────────────────────


class TestPITSafetyNet:
    """PIT safety net в `load_prices`: warning при |pct_change| > 20%."""

    def test_detects_anomalous_returns(self, caplog):
        """Цены с синтетическим gap (>20% в день) → warning с деталями."""
        import logging

        import pandas as pd

        from aqr.tools.core import _pit_check_anomalous_returns

        # Серия с двумя нормальными днями и одним синтетическим gap (+50%)
        series = pd.Series(
            [100.0, 101.0, 151.5, 152.0, 153.0],  # +50% на индексе 2
            index=pd.date_range("2024-01-01", periods=5),
        )
        with caplog.at_level(logging.WARNING, logger="aqr.tools.core"):
            _pit_check_anomalous_returns({"SBER": series})
        # Должен быть хотя бы один warning про SBER
        warnings = [r for r in caplog.records if "SBER" in r.message]
        assert len(warnings) >= 1, f"No PIT warnings logged: {[r.message for r in caplog.records]}"
        assert "20%" in warnings[0].message

    def test_silent_for_clean_series(self, caplog):
        """Нормальные движения (~1-3%) — без warning."""
        import logging

        import pandas as pd

        from aqr.tools.core import _pit_check_anomalous_returns

        # Плавные движения в пределах 5%
        series = pd.Series(
            [100.0, 102.0, 101.5, 103.0, 102.5],
            index=pd.date_range("2024-01-01", periods=5),
        )
        with caplog.at_level(logging.WARNING, logger="aqr.tools.core"):
            _pit_check_anomalous_returns({"SBER": series})
        assert not any("SBER" in r.message for r in caplog.records)

    def test_silent_for_short_series(self, caplog):
        """Менее 2 точек → pct_change даёт пустой Series → warning не нужен."""
        import logging

        import pandas as pd

        from aqr.tools.core import _pit_check_anomalous_returns

        with caplog.at_level(logging.WARNING, logger="aqr.tools.core"):
            _pit_check_anomalous_returns({"SBER": pd.Series([100.0])})
        assert not any("SBER" in r.message for r in caplog.records)


# ── generate_hypotheses_tool ─────────────────────────────────────

class TestGenerateHypotheses:
    @pytest.mark.asyncio
    async def test_generates_n_hypotheses(self):
        tool = registry.get("generate_hypotheses")
        result = await tool.fn(
            tickers=["SBER", "GAZP"],
            families=["momentum", "mean_reversion"],
            n=10,
        )
        assert len(result) == 10
        for h in result:
            assert "name" in h
            assert "family" in h
            assert "ticker" in h
            assert "params" in h

    @pytest.mark.asyncio
    async def test_all_hypotheses_have_known_family(self):
        tool = registry.get("generate_hypotheses")
        result = await tool.fn(
            tickers=["SBER"],
            families=["momentum", "breakout"],
            n=6,
        )
        valid_families = {"momentum", "mean_reversion", "breakout", "volatility"}
        for h in result:
            assert h["family"] in valid_families

    @pytest.mark.asyncio
    async def test_empty_tickers_returns_empty(self):
        tool = registry.get("generate_hypotheses")
        result = await tool.fn(tickers=[], families=["momentum"], n=5)
        assert len(result) == 0


# ── backtest_one ─────────────────────────────────────────────────

class TestBacktestOne:
    @pytest.mark.asyncio
    async def test_backtest_returns_metrics(self, sample_hypothesis, synthetic_prices):
        tool = registry.get("backtest_one")
        result = await tool.fn(
            hypothesis=sample_hypothesis,
            prices=synthetic_prices,
        )
        assert "error" not in result
        assert "sharpe" in result
        assert "dsr" in result
        assert "dsr_verdict" in result
        assert "max_drawdown" in result
        assert "n_trades" in result
        assert "daily_returns" in result

    @pytest.mark.asyncio
    async def test_backtest_has_daily_returns(self, sample_hypothesis, synthetic_prices):
        """daily_returns нужны для PBO-валидации."""
        tool = registry.get("backtest_one")
        result = await tool.fn(
            hypothesis=sample_hypothesis,
            prices=synthetic_prices,
        )
        assert len(result["daily_returns"]) > 0

    @pytest.mark.asyncio
    async def test_backtest_unknown_family(self, synthetic_prices):
        tool = registry.get("backtest_one")
        result = await tool.fn(
            hypothesis={"family": "nonexistent", "ticker": "XXX", "params": {}},
            prices=synthetic_prices,
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_backtest_insufficient_data(self, synthetic_prices):
        """Слишком короткий ряд — insufficient."""
        tool = registry.get("backtest_one")
        result = await tool.fn(
            hypothesis={
                "name": "Test", "family": "momentum",
                "ticker": "T", "params": {"fast": 5, "slow": 20},
            },
            prices=synthetic_prices[:10],  # всего 10 баров
        )
        assert result["dsr_verdict"] == "insufficient"
        assert result["n_trades"] == 0

    @pytest.mark.asyncio
    async def test_backtest_accepts_cpcv_config_override(self, synthetic_prices):
        """cpcv_splits / embargo_pct из kwargs пробрасываются."""
        tool = registry.get("backtest_one")
        result = await tool.fn(
            hypothesis={
                "name": "T", "family": "momentum",
                "ticker": "SBER", "params": {"fast": 10, "slow": 50},
            },
            prices=synthetic_prices,
            cpcv_splits=4,
            cpcv_test_splits=1,
            embargo_pct=0.05,
        )
        # Не падает и возвращает корректные cpcv-метрики
        assert "cpcv_mean_sharpe" in result
        assert "cpcv_std_sharpe" in result


# ── validate_portfolio ──────────────────────────────────────────

class TestValidatePortfolio:
    @pytest.mark.asyncio
    async def test_insufficient_with_few_results(self):
        """Меньше 4 результатов — insufficient."""
        tool = registry.get("validate_portfolio")
        result = await tool.fn(results=[{"daily_returns": [0.01, -0.02, 0.01]}])
        assert result["verdict"] == "insufficient"

    @pytest.mark.asyncio
    async def test_pbo_with_backtest_results(self, sample_hypothesis, synthetic_prices):
        """Полный цикл: backtest → validate."""
        bt_tool = registry.get("backtest_one")
        val_tool = registry.get("validate_portfolio")

        results = []
        for seed in range(6):
            px = (np.array(synthetic_prices) * (1 + seed * 0.001)).tolist()
            h = {**sample_hypothesis, "ticker": f"T{seed}"}
            r = await bt_tool.fn(hypothesis=h, prices=px)
            if "error" not in r:
                results.append(r)

        if len(results) >= 4:
            pbo = await val_tool.fn(results=results)
            assert "pbo" in pbo
            assert "verdict" in pbo
            assert isinstance(pbo["pbo"], float)


# ── extract_insights ────────────────────────────────────────────

class TestExtractInsights:
    @pytest.mark.asyncio
    async def test_empty_top_returns_empty(self):
        tool = registry.get("extract_insights")
        result = await tool.fn(
            top_results=[], n_tested=10, n_survived=5, pbo=0.3, pbo_verdict="ok",
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_extract_returns_observations(self):
        tool = registry.get("extract_insights")
        top = [
            {
                "name": "SMA10/50", "family": "momentum", "ticker": "SBER",
                "sharpe": 1.4, "dsr": 0.97, "dsr_verdict": "significant",
                "cpcv_mean_sharpe": 0.8, "cpcv_std_sharpe": 0.2,
                "max_drawdown": -0.12, "n_trades": 50, "params": {"fast": 10, "slow": 50},
            },
            {
                "name": "BB-mean-reversion", "family": "mean_reversion", "ticker": "GAZP",
                "sharpe": 0.7, "dsr": 0.82, "dsr_verdict": "borderline",
                "cpcv_mean_sharpe": 0.4, "cpcv_std_sharpe": 0.3,
                "max_drawdown": -0.18, "n_trades": 32, "params": {"window": 20},
            },
        ]
        result = await tool.fn(
            top_results=top, n_tested=20, n_survived=8,
            pbo=0.35, pbo_verdict="borderline",
        )
        assert isinstance(result, list)
        assert len(result) >= 3
        joined = " ".join(result).lower()
        assert "sma10/50" in joined
        assert "momentum" in joined
        assert "mean_reversion" in joined
        assert "pbo" in joined or "переобучение" in joined or "перепроверь" in joined
        assert "8/20" in joined or "40%" in joined

    @pytest.mark.asyncio
    async def test_high_pbo_warns_about_overfitting(self):
        tool = registry.get("extract_insights")
        top = [{
            "name": "X", "family": "momentum", "ticker": "SBER",
            "sharpe": 1.0, "dsr": 0.9, "dsr_verdict": "significant",
            "max_drawdown": -0.1, "n_trades": 10, "params": {},
        }]
        result = await tool.fn(
            top_results=top, n_tested=10, n_survived=2,
            pbo=0.7, pbo_verdict="high",
        )
        assert any("переобучен" in s.lower() or "внимание" in s.lower() for s in result)

# ── review_insights ─────────────────────────────────────────────

class TestReviewInsights:
    @pytest.mark.asyncio
    async def test_review_returns_list(
        self, monkeypatch, with_credentials, fake_litellm
    ):
        """С credentials + мок-LLM → список observations."""
        fake_litellm('{"observations": ["тестовый инсайт"]}')
        monkeypatch.setenv("AQR_LLM_MODEL", "claude-3-5-sonnet-20241022")

        tool = registry.get("review_insights")
        result = await tool.fn(
            goal="проверь momentum",
            top_results=[{
                "name": "SMA10/50", "family": "momentum", "ticker": "SBER",
                "sharpe": 1.2, "dsr": 0.9, "dsr_verdict": "significant",
                "max_drawdown": -0.15, "n_trades": 42, "params": {},
                "cpcv_mean_sharpe": 1.0,
            }],
            deterministic_insights=["тестовый инсайт"],
        )
        assert isinstance(result, list)


# ── narrate ─────────────────────────────────────────────────────

class TestNarrate:
    @pytest.mark.asyncio
    async def test_narrate_returns_string(
        self, monkeypatch, with_credentials, fake_litellm
    ):
        """С credentials + мок-LLM → текст нарратива."""
        fake_litellm("Тестовый нарратив от LLM про momentum стратегию.")
        monkeypatch.setenv("AQR_LLM_MODEL", "claude-3-5-sonnet-20241022")

        tool = registry.get("narrate")
        result = await tool.fn(
            goal="проверь momentum на Сбере",
            tickers=["SBER"],
            families=["momentum"],
            n_tested=20,
            n_survived=5,
            pbo=0.3,
            pbo_verdict="ok",
            top_results=[{
                "name": "SMA10/50", "family": "momentum", "ticker": "SBER",
                "sharpe": 1.2, "dsr": 0.9, "dsr_verdict": "significant",
                "max_drawdown": -0.15, "n_trades": 42,
                "params": {"fast": 10, "slow": 50},
            }],
            elapsed_seconds=2.5,
        )
        assert isinstance(result, str)
        assert len(result) > 10
