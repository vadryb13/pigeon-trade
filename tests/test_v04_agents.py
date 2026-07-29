"""Tests for v0.4 5-agent team: individual agents + orchestrator e2e.

Mock strategy:
  - Individual agents: mock tool registry lookups + external deps
  - Orchestrator: mock agent class methods (plan, analyze, validate, write)
"""
from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from aqr.graph.context import reset_credentials, set_credentials
from aqr.registry import DecryptedSettings

# ── Helpers ──────────────────────────────────────────────────────

def _register_tool(name: str, return_value, registry, mocks: dict):
    """Register a tool with a controllable mock.

    The mock is stored in `mocks[name]` AND in the registry ToolSpec.
    Replaces any existing tool with the same name.
    """
    mock = AsyncMock(return_value=return_value)
    mocks[name] = mock
    from aqr.tools import ToolSpec
    spec = ToolSpec(name=name, description="", input_schema={}, fn=mock)
    existing = registry.get(name)
    if existing is not None:
        registry._tools[name] = spec
    else:
        registry.register(spec)
    return mock


def _set_tool_side_effect(name: str, side_effect, registry, mocks: dict):
    """Replace a tool's mock with one that raises `side_effect`.

    Updates both the mocks dict and the registry ToolSpec.fn.
    """
    mock = AsyncMock(side_effect=side_effect)
    mocks[name] = mock
    tool = registry.get(name)
    if tool is not None:
        tool.fn = mock


# ── Base fixtures ────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _stable_secret(monkeypatch):
    monkeypatch.setenv("AQR_SESSION_SECRET", "test-secret-padded-to-32-bytes-base64==")


@pytest.fixture(autouse=True)
def _clear_registry(monkeypatch):
    """Reset the global tool registry before each test."""
    from aqr.tools import reset_for_testing
    reset_for_testing()


@pytest.fixture
def with_credentials():
    """Set session credentials into ContextVar."""
    creds = DecryptedSettings(
        session_id="test-session",
        llm_model="claude-3-5-sonnet-20241022",
        llm_api_key="sk-ant-fake",
        openai_api_key="sk-oai-fake",
        invest_token="t.INVEST_TOKEN_fake",
        invest_sandbox=True,
    )
    token = set_credentials(creds)
    yield creds
    reset_credentials(token)


class _FakeToolRegistry(dict):
    """Dict that also exposes `set_side_effect(name, exc)` for tests."""

    registry_ref = None  # set by fixture

    def set_side_effect(self, name: str, exc):
        _set_tool_side_effect(name, exc, self.registry_ref, self)


@pytest.fixture
def fake_tool_registry():
    """Register fake tools into the global registry.

    Returns a _FakeToolRegistry dict {name: AsyncMock} for assertions,
    plus `set_side_effect(name, exc)` helper.
    """
    from aqr.tools import registry

    mocks = _FakeToolRegistry()
    mocks.registry_ref = registry

    _register_tool("plan_research", {
        "tickers": ["SBER"],
        "hypothesis_families": ["momentum"],
        "n_hypotheses": 10,
        "start_date": "2023-01-01",
        "end_date": "2024-12-31",
        "timeframe": "D1",
        "rationale": "test",
        "observations": [],
    }, registry, mocks)

    _register_tool("load_prices", {
        "SBER": [100.0 + i * 0.1 for i in range(500)],
    }, registry, mocks)

    _register_tool("backtest_one", {
        "name": "momentum_SBER",
        "family": "momentum",
        "ticker": "SBER",
        "params": {"fast": 10, "slow": 30},
        "sharpe": 1.5,
        "dsr": 1.2,
        "dsr_verdict": "significant",
        "cpcv_mean_sharpe": 0.9,
        "cpcv_std_sharpe": 0.3,
        "max_drawdown": -0.15,
        "n_trades": 42,
        "daily_returns": [0.001] * 300,
    }, registry, mocks)

    _register_tool("validate_portfolio", {
        "pbo": 0.32,
        "verdict": "ok",
    }, registry, mocks)

    _register_tool("extract_insights", [
        "Лучшая гипотеза: momentum/SBER (DSR=1.20)",
        "PBO=0.32 — отбор в OOS выглядит устойчивым.",
    ], registry, mocks)

    _register_tool("narrate", "Тестовый отчёт о momentum на Сбере.", registry, mocks)

    return mocks


@pytest.fixture
def mock_openai(monkeypatch):
    """Mock openai.AsyncOpenAI for Embedder."""
    class _FakeEmbAPI:
        async def create(self, *, model, input):
            return MagicMock(data=[MagicMock(embedding=[0.1] * 1536)])

    class _FakeAIOpenAI:
        def __init__(self, **kw):
            self.embeddings = _FakeEmbAPI()

    fake = types.ModuleType("openai")
    fake.AsyncOpenAI = _FakeAIOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake)


@pytest.fixture
def mock_db(monkeypatch):
    """Mock async_session_factory for DB-dependent agents."""
    from aqr import session as db_mod

    class _FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

        async def execute(self, *a, **kw):
            class _R:
                def scalars(self):
                    return self
                def all(self):
                    return []
                def scalar(self):
                    return None
            return _R()

        async def get(self, *a, **kw):
            return None

        async def commit(self):
            return None

        async def flush(self):
            return None

        def add(self, *a, **kw):
            return None

    monkeypatch.setattr(db_mod, "async_session_factory", lambda: _FakeSession())


@pytest.fixture
def mock_tinvest(monkeypatch):
    """Mock TInvestAdapter for BrowserAgent."""
    from aqr.data import tinvest as tinvest_mod

    class _FakeAdapter:
        def __init__(self, *a, **kw):
            pass

        async def _resolve_figi(self, ticker):
            return {"SBER": "BBG004730N88", "GAZP": "BBG004730RP0"}.get(ticker, "")

    monkeypatch.setattr(tinvest_mod, "TInvestAdapter", _FakeAdapter)


# ── EditorAgent tests ────────────────────────────────────────────

class TestEditorAgent:
    @pytest.mark.asyncio
    async def test_plan_returns_plan_dict(self, with_credentials, fake_tool_registry):
        from aqr.agents.editor import EditorAgent
        agent = EditorAgent("test-session")
        result = await agent.plan("проверь momentum на Сбере")
        assert result.ok
        plan = result.data["plan"]
        assert "SBER" in plan.get("tickers", [])
        assert "momentum" in plan.get("hypothesis_families", [])
        assert plan.get("n_hypotheses") == 10

    @pytest.mark.asyncio
    async def test_plan_uses_tool_registry(self, with_credentials, fake_tool_registry):
        from aqr.agents.editor import EditorAgent
        agent = EditorAgent("test-session")
        await agent.plan("тест")
        fake_tool_registry["plan_research"].assert_awaited_once_with(goal="тест")

    @pytest.mark.asyncio
    async def test_plan_fallback_when_tool_missing(self, with_credentials):
        """Without plan_research, Editor returns AgentResult(ok=False) with error message."""
        from aqr.tools import registry as tool_registry
        # Симулируем отсутствие инструмента — удаляем из registry
        plan_tool = tool_registry.get("plan_research")
        if plan_tool is not None:
            tool_registry._tools.pop("plan_research", None)

        from aqr.agents.editor import EditorAgent
        agent = EditorAgent("test-session")
        result = await agent.plan("тест")
        assert not result.ok
        assert "plan_research tool is not registered" in result.error

    @pytest.mark.asyncio
    async def test_plan_without_credentials_raises(self):
        """Without credentials, BaseAgent.credentials raises RuntimeError."""
        from aqr.agents.editor import EditorAgent
        agent = EditorAgent("test-session")
        with pytest.raises(RuntimeError, match="not configured"):
            _ = agent.credentials


# ── BrowserAgent tests ───────────────────────────────────────────

class TestBrowserAgent:
    @pytest.mark.asyncio
    async def test_research_returns_context(self, with_credentials, mock_db, mock_openai, mock_tinvest):
        from aqr.agents.browser import BrowserAgent
        agent = BrowserAgent("test-session")
        result = await agent.research("тест", {"tickers": ["SBER"]})
        assert result.ok
        assert "similar_runs" in result.data
        assert "ticker_info" in result.data

    @pytest.mark.asyncio
    async def test_research_ticker_info(self, with_credentials, mock_db, mock_openai, mock_tinvest):
        from aqr.agents.browser import BrowserAgent
        agent = BrowserAgent("test-session")
        result = await agent.research("тест", {"tickers": ["SBER", "GAZP"]})
        info = result.data["ticker_info"]
        assert "SBER" in info
        assert info["SBER"]["figi"] == "BBG004730N88"

    @pytest.mark.asyncio
    async def test_research_empty_goal(self, with_credentials, mock_db, mock_openai):
        from aqr.agents.browser import BrowserAgent
        agent = BrowserAgent("test-session")
        result = await agent.research("", {})
        assert result.ok
        assert result.data["similar_runs"] == []

    @pytest.mark.asyncio
    async def test_research_without_credentials_raises(self):
        from aqr.agents.browser import BrowserAgent
        agent = BrowserAgent("test-session")
        with pytest.raises(RuntimeError, match="not configured"):
            _ = agent.credentials


# ── AnalystAgent tests ───────────────────────────────────────────

class TestAnalystAgent:
    @pytest.mark.asyncio
    async def test_analyze_returns_results(self, with_credentials, fake_tool_registry):
        from aqr.agents.analyst import AnalystAgent
        agent = AnalystAgent("test-session")
        result = await agent.analyze("SBER", ["momentum", "mean_reversion"])
        assert result.ok
        assert result.data["ticker"] == "SBER"
        assert len(result.data["results"]) > 0
        fake_tool_registry["load_prices"].assert_awaited()
        fake_tool_registry["backtest_one"].assert_awaited()

    @pytest.mark.asyncio
    async def test_analyze_load_prices_fail(self, with_credentials, fake_tool_registry):
        fake_tool_registry.set_side_effect("load_prices", ValueError("no data"))
        from aqr.agents.analyst import AnalystAgent
        agent = AnalystAgent("test-session")
        result = await agent.analyze("SBER", ["momentum"])
        assert not result.ok
        assert "no data" in result.error

    @pytest.mark.asyncio
    async def test_analyze_insufficient_prices(self, with_credentials, fake_tool_registry):
        fake_tool_registry.set_side_effect("load_prices", ValueError("SBER: [100.0] * 50"))
        from aqr.agents.analyst import AnalystAgent
        agent = AnalystAgent("test-session")
        result = await agent.analyze("SBER", ["momentum"])
        assert not result.ok
        assert "insufficient data" in result.error or "load_prices failed" in result.error

    @pytest.mark.asyncio
    async def test_analyze_non_momentum(self, with_credentials, fake_tool_registry):
        """Non-momentum families generate hypotheses and run deep backtest."""
        from aqr.agents.analyst import AnalystAgent
        agent = AnalystAgent("test-session")
        result = await agent.analyze("SBER", ["mean_reversion"])
        assert result.ok
        assert result.data["n_deep"] > 0
        fake_tool_registry["backtest_one"].assert_awaited()


# ── ReviewerAgent tests ──────────────────────────────────────────

class TestReviewerAgent:
    @pytest.mark.asyncio
    async def test_validate_returns_metrics(self, with_credentials, fake_tool_registry):
        from aqr.agents.reviewer import ReviewerAgent
        agent = ReviewerAgent("test-session")
        results = [
            {"dsr_verdict": "significant", "sharpe": 1.5, "dsr": 1.2},
            {"dsr_verdict": "borderline", "sharpe": 0.9, "dsr": 0.85},
            {"dsr_verdict": "not_significant", "sharpe": 0.3, "dsr": 0.2},
        ]
        result = await agent.validate(results)
        assert result.ok
        assert result.data["n_tested"] == 3
        assert result.data["n_survived"] == 2
        assert "best_result" in result.data
        fake_tool_registry["validate_portfolio"].assert_awaited()

    @pytest.mark.asyncio
    async def test_validate_empty_results(self, with_credentials, fake_tool_registry):
        from aqr.agents.reviewer import ReviewerAgent
        agent = ReviewerAgent("test-session")
        result = await agent.validate([])
        assert result.ok
        assert result.data["n_tested"] == 0
        assert result.data["n_survived"] == 0
        assert "Нет результатов" in result.data["recommendations"][0]

    @pytest.mark.asyncio
    async def test_validate_single_result(self, with_credentials, fake_tool_registry):
        from aqr.agents.reviewer import ReviewerAgent
        agent = ReviewerAgent("test-session")
        results = [{"dsr_verdict": "significant", "sharpe": 2.0, "dsr": 1.8}]
        result = await agent.validate(results)
        assert result.ok
        assert result.data["n_tested"] == 1
        assert result.data["n_survived"] == 1
        assert result.data["best_result"]["dsr"] == 1.8


# ── WriterAgent tests ────────────────────────────────────────────

class TestWriterAgent:
    @pytest.mark.asyncio
    async def test_write_returns_narrative(self, with_credentials, fake_tool_registry):
        from aqr.agents.writer import WriterAgent
        agent = WriterAgent("test-session")
        result = await agent.write(
            goal="тест",
            plan={"tickers": ["SBER"]},
            all_results=[{"sharpe": 1.5, "dsr": 1.2, "dsr_verdict": "significant",
                          "family": "momentum", "ticker": "SBER", "params": {"fast": 10}}],
            validation={"pbo": 0.32, "pbo_verdict": "ok", "n_survived": 1,
                        "n_tested": 3, "best_result": None, "aggregate": {}},
        )
        assert result.ok
        assert result.data["narrative"]
        assert "Тестовый отчёт" in result.data["narrative"]
        assert len(result.data["insights"]) > 0

    @pytest.mark.asyncio
    async def test_write_empty_results(self, with_credentials):
        from aqr.agents.writer import WriterAgent
        agent = WriterAgent("test-session")
        result = await agent.write(goal="test", all_results=[])
        assert result.ok
        assert "не удалось" in result.data["narrative"].lower()
        assert result.data["top_results"] == []

    @pytest.mark.asyncio
    async def test_write_top_dsr_sorting(self, with_credentials, fake_tool_registry):
        from aqr.agents.writer import WriterAgent
        agent = WriterAgent("test-session")
        results = [
            {"dsr": 0.5, "sharpe": 1.0, "dsr_verdict": "borderline",
             "family": "a", "ticker": "SBER", "params": {}},
            {"dsr": 1.5, "sharpe": 2.0, "dsr_verdict": "significant",
             "family": "b", "ticker": "SBER", "params": {}},
        ]
        result = await agent.write(
            goal="test",
            all_results=results,
            validation={"n_survived": 1, "pbo": 0.3, "pbo_verdict": "ok",
                        "n_tested": 2, "best_result": None, "aggregate": {}},
        )
        assert result.ok
        top = result.data["top_results"]
        assert len(top) == 2
        # Sorted by DSR desc
        assert top[0]["dsr"] == 1.5


# ── Orchestrator e2e tests ───────────────────────────────────────

class TestRunTeam:
    @pytest.mark.asyncio
    async def test_run_team_e2e(self, with_credentials, fake_tool_registry, mock_db, mock_openai):
        """Full team run with all mocked tools returns TeamResult with narrative."""
        from aqr.agents.orchestrator import run_team
        result = await run_team(
            goal="проверь momentum на Сбере",
            session_id="test-e2e",
        )
        assert result.ok
        assert result.goal == "проверь momentum на Сбере"
        assert result.narrative
        assert len(result.insights) > 0
        assert result.summary
        assert result.elapsed_seconds > 0
        assert result.n_tested > 0
        assert "SBER" in str(result.plan)

    @pytest.mark.asyncio
    async def test_run_team_with_ticker_override(self, with_credentials, fake_tool_registry, mock_db, mock_openai):
        """Ticker override bypasses Editor's ticker list."""
        from aqr.agents.orchestrator import run_team
        result = await run_team(
            goal="тест",
            session_id="test-override",
            tickers=["GAZP"],
            families=["momentum"],
        )
        assert result.ok
        assert "GAZP" in str(result.plan)

    @pytest.mark.asyncio
    async def test_run_team_error_propagation(self, with_credentials, fake_tool_registry):
        """If Editor plan_research fails, error propagates as TeamResult(ok=False)."""
        from aqr.tools import registry as tool_registry
        tool = tool_registry.get("plan_research")
        tool.fn = AsyncMock(side_effect=RuntimeError("LLM down"))
        from aqr.agents.orchestrator import run_team
        result = await run_team("тест")
        assert not result.ok
        assert "LLM down" in result.error

    @pytest.mark.asyncio
    async def test_run_team_with_explicit_goal(self, with_credentials, fake_tool_registry, mock_db, mock_openai):
        """run_team returns goal as provided."""
        from aqr.agents.orchestrator import run_team
        result = await run_team("проверь mean reversion на GAZP", "test-explicit")
        assert result.goal == "проверь mean reversion на GAZP"

    @pytest.mark.asyncio
    async def test_run_team_analyst_uses_backtest_tool(self, with_credentials, fake_tool_registry, mock_db, mock_openai):
        """Verify that backtest_one tool is called during team run."""
        from aqr.agents.orchestrator import run_team
        await run_team("тест", "test-bt-call")
        fake_tool_registry["load_prices"].assert_awaited()
        fake_tool_registry["backtest_one"].assert_awaited()
        fake_tool_registry["validate_portfolio"].assert_awaited()
        fake_tool_registry["narrate"].assert_awaited()

    @pytest.mark.asyncio
    async def test_run_team_with_agent_errors(self, with_credentials, fake_tool_registry, mock_db, mock_openai):
        """TeamResult collects agent errors without crashing."""
        fake_tool_registry.set_side_effect("load_prices", ValueError("no data"))
        from aqr.agents.orchestrator import run_team

        # Analyst will fail (no prices) -> result should still be returned
        result = await run_team("тест", "test-errors")
        assert result.ok
        assert result.agent_errors  # at least one error collected
