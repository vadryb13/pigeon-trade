"""Unit tests for PipelineExecutor — critical execution path."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from aqr.pipeline.events import Event, EventBus
from aqr.pipeline.executor import PipelineExecutor
from aqr.pipeline.planner import ResearchPlan


@pytest.fixture(autouse=True)
def _restore_registry():
    """Restore the real tool registry after tests that mutate it."""
    from aqr.tools import registry

    saved = dict(registry._tools)
    yield
    registry._tools = saved
    import aqr.tools.register
    aqr.tools.register._registration_done = True


def _make_plan(**overrides):
    return ResearchPlan(
        goal="тест",
        tickers=overrides.get("tickers", ["SBER"]),
        start_date=overrides.get("start_date", "2023-01-01"),
        end_date=overrides.get("end_date", "2024-12-31"),
        timeframe=overrides.get("timeframe", "D1"),
        hypothesis_families=overrides.get("hypothesis_families", ["momentum"]),
        n_hypotheses=overrides.get("n_hypotheses", 8),
        validation=overrides.get("validation", {}),
    )

def _fake_tool_registry_with_defaults(monkeypatch, **overrides):
    """Устанавливает моки для всех инструментов через monkeypatch."""
    from aqr.tools import ToolSpec, registry

    monkeypatch.setattr("aqr.tools.register.register_all", lambda: None)

    registry._tools = {}
    import aqr.tools.register
    aqr.tools.register._registration_done = False

    _empty_schema = {"type": "object", "properties": {}}

    async def _load_prices_fn(**kw):
        return {"SBER": [100.0 + i * 0.5 for i in range(500)]}

    async def _gen_fn(**kw):
        return [{"name": "SMA5/20", "family": "momentum", "ticker": "SBER",
                 "params": {"fast": 5, "slow": 20}}]

    bt_result = {
        "name": "SMA5/20", "family": "momentum", "ticker": "SBER",
        "params": {"fast": 5, "slow": 20},
        "sharpe": 1.2, "dsr": 0.9, "dsr_verdict": "significant",
        "cpcv_mean_sharpe": 1.0, "cpcv_std_sharpe": 0.2,
        "max_drawdown": -0.1, "n_trades": 42, "daily_returns": [0.001] * 50,
    }

    async def _bt_fn(**kw):
        return bt_result

    async def _val_fn(**kw):
        return {"pbo": 0.3, "verdict": "ok"}

    async def _ins_fn(**kw):
        return ["Шарп 1.2 — хорошо"]

    async def _rev_fn(**kw):
        return ["LLM: отличный сигнал"]

    async def _nar_fn(**kw):
        return "Итоговый нарратив"

    defs = {
        "load_prices": overrides.get("load_prices", _load_prices_fn),
        "generate_hypotheses": overrides.get("generate_hypotheses", _gen_fn),
        "backtest_one": overrides.get("backtest_one", _bt_fn),
        "validate_portfolio": overrides.get("validate_portfolio", _val_fn),
        "extract_insights": overrides.get("extract_insights", _ins_fn),
        "review_insights": overrides.get("review_insights", _rev_fn),
        "narrate": overrides.get("narrate", _nar_fn),
    }

    for name, fn in defs.items():
        registry._tools[name] = ToolSpec(
            name=name, description="mock", input_schema=_empty_schema, fn=fn,
        )


# ---- Tests ----

class TestPipelineExecutorRun:
    @pytest.mark.asyncio
    async def test_full_pipeline_emits_done_event(self, monkeypatch):
        """Полный пайплайн завершается событием 'done'."""
        _fake_tool_registry_with_defaults(monkeypatch)
        bus = EventBus()
        executor = PipelineExecutor(bus)
        plan = _make_plan()

        result = await executor.run("run-1", plan)

        assert result.n_hypotheses_tested == 1
        assert result.n_survived_dsr == 1
        assert result.portfolio_pbo == 0.3
        assert len(result.top) == 1
        events = bus.history("run-1")
        kinds = [e.kind for e in events]
        assert "done" in kinds
        assert "backtesting" in kinds
        assert "insight" in kinds

    @pytest.mark.asyncio
    async def test_skips_missing_ticker_in_prices(self, monkeypatch):
        """Гипотеза для тикера без цен — пропускается."""
        async def _gazp_only(**kw):
            return {"GAZP": [100.0 + i * 0.5 for i in range(500)]}

        _fake_tool_registry_with_defaults(monkeypatch, load_prices=_gazp_only)
        bus = EventBus()
        executor = PipelineExecutor(bus)
        plan = _make_plan(tickers=["SBER"])

        result = await executor.run("run-2", plan)

        assert result.n_hypotheses_tested == 0
        assert result.n_survived_dsr == 0
        assert len(result.top) == 0

    @pytest.mark.asyncio
    async def test_validation_config_forwards_to_backtest(self, monkeypatch):
        """CPCV-конфиг из плана пробрасывается в backtest_one."""
        bt_mock = AsyncMock(return_value={
            "name": "SMA5/20", "family": "momentum", "ticker": "SBER",
            "params": {}, "sharpe": 0.5, "dsr": 0.3, "dsr_verdict": "borderline",
            "cpcv_mean_sharpe": 0.4, "cpcv_std_sharpe": 0.1,
            "max_drawdown": -0.05, "n_trades": 10, "daily_returns": [0.001] * 30,
        })
        _fake_tool_registry_with_defaults(monkeypatch, backtest_one=bt_mock)
        bus = EventBus()
        executor = PipelineExecutor(bus)
        plan = _make_plan(validation={"cpcv_splits": 8, "cpcv_test_splits": 3, "embargo_pct": 0.05})

        await executor.run("run-3", plan)

        call_kwargs = bt_mock.call_args.kwargs
        assert call_kwargs["cpcv_splits"] == 8
        assert call_kwargs["cpcv_test_splits"] == 3
        assert call_kwargs["embargo_pct"] == 0.05

    @pytest.mark.asyncio
    async def test_dict_to_backtest_result_defaults(self):
        """_dict_to_backtest_result заполняет дефолты для отсутствующих полей."""
        result = PipelineExecutor._dict_to_backtest_result({})
        assert result.sharpe == 0
        assert result.dsr == 0
        assert result.dsr_verdict == "?"
        assert result.n_trades == 0
        assert result.daily_returns == []
        assert result.hypothesis.name == "?"

    @pytest.mark.asyncio
    async def test_review_insights_failure_emits_warning(self, monkeypatch):
        """Сбой review_insights → warning-событие, пайплайн не падает."""
        rev = MagicMock()
        rev.fn = AsyncMock(side_effect=RuntimeError("LLM timeout"))
        _fake_tool_registry_with_defaults(monkeypatch, review_insights=rev)
        bus = EventBus()
        executor = PipelineExecutor(bus)
        plan = _make_plan()

        result = await executor.run("run-4", plan)

        events = bus.history("run-4")
        kinds = [e.kind for e in events]
        assert "warning" in kinds
        assert "done" in kinds
        assert result.n_hypotheses_tested == 1

    @pytest.mark.asyncio
    async def test_narrate_failure_emits_error_and_raises(self, monkeypatch):
        """Сбой narrate → error-событие + raise."""
        async def _failing_narrate(**kw):
            raise RuntimeError("LLM quota exceeded")

        _fake_tool_registry_with_defaults(monkeypatch, narrate=_failing_narrate)
        bus = EventBus()
        executor = PipelineExecutor(bus)
        plan = _make_plan()

        with pytest.raises(RuntimeError, match="LLM quota exceeded"):
            await executor.run("run-5", plan)

        events = bus.history("run-5")
        kinds = [e.kind for e in events]
        assert "error" in kinds

    @pytest.mark.asyncio
    async def test_backtest_error_dict_skipped(self, monkeypatch):
        """backtest_one вернул error dict → результат пропускается."""
        bt = AsyncMock(return_value={"error": "no data"})
        _fake_tool_registry_with_defaults(monkeypatch, backtest_one=bt)
        bus = EventBus()
        executor = PipelineExecutor(bus)
        plan = _make_plan()

        result = await executor.run("run-6", plan)

        assert result.n_hypotheses_tested == 0
        assert len(result.top) == 0


class TestEventBusQueueFull:
    @pytest.mark.asyncio
    async def test_queue_full_drops_event_and_logs(self):
        """Переполненная очередь дропает события без падения."""
        bus = EventBus()
        tiny_q = asyncio.Queue(maxsize=1)
        tiny_q.put_nowait = MagicMock(side_effect=asyncio.QueueFull)
        bus._subscribers["run-1"] = [tiny_q]

        ev = Event(run_id="run-1", kind="backtesting", stage="test", message="msg")

        await bus.publish(ev)

        assert len(bus.history("run-1")) == 1
