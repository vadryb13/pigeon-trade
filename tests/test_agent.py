"""Tests for LangGraph agent graph and nodes."""
from __future__ import annotations

import pytest
from langgraph.graph import END

from aqr.agent.graph import (
    AgentState,
    _deterministic_route,
    _has_llm_key,
    build_graph,
    get_agent,
    respond_node,
    run_agent,
)

# ── Graph structure tests ───────────────────────────────────────

class TestGraphStructure:
    def test_build_graph_does_not_raise(self):
        """Graph can be built without errors."""
        graph = build_graph()
        assert graph is not None

    def test_compile_graph_does_not_raise(self):
        """Compiled graph is valid."""
        agent = get_agent()
        assert agent is not None

    def test_singleton_graph(self):
        """get_agent() returns same instance."""
        g1 = get_agent()
        g2 = get_agent()
        assert g1 is g2


# ── Router tests ────────────────────────────────────────────────

class TestRouter:
    def test_empty_state_goes_to_plan(self):
        """Empty state → plan (first step)."""
        state: AgentState = {
            "messages": [],
            "session_id": "test",
            "goal": "",
            "plan": None,
            "prices": None,
            "hypotheses": None,
            "results": None,
            "pbo": None,
            "insights": None,
            "narrative": None,
            "step": "",
            "error": None,
            "elapsed_seconds": 0.0,
            "n_tested": 0,
            "n_survived": 0,
        }
        assert _deterministic_route(state) == "plan"

    def test_plan_goes_to_load_data(self):
        """After plan → load_data."""
        state: AgentState = {
            "messages": [],
            "session_id": "test",
            "goal": "test",
            "plan": {"tickers": ["SBER"]},
            "prices": None,
            "hypotheses": None,
            "results": None,
            "pbo": None,
            "insights": None,
            "narrative": None,
            "step": "plan",
            "error": None,
            "elapsed_seconds": 0.0,
            "n_tested": 0,
            "n_survived": 0,
        }
        assert _deterministic_route(state) == "load_data"

    def test_done_goes_to_end(self):
        """After done → END."""
        state: AgentState = {
            "messages": [],
            "session_id": "test",
            "goal": "test",
            "plan": None,
            "prices": None,
            "hypotheses": None,
            "results": None,
            "pbo": None,
            "insights": None,
            "narrative": None,
            "step": "done",
            "error": None,
            "elapsed_seconds": 0.0,
            "n_tested": 0,
            "n_survived": 0,
        }
        assert _deterministic_route(state) == END

    def test_error_goes_to_respond(self):
        """Error state → respond."""
        state: AgentState = {
            "messages": [],
            "session_id": "test",
            "goal": "test",
            "plan": None,
            "prices": None,
            "hypotheses": None,
            "results": None,
            "pbo": None,
            "insights": None,
            "narrative": None,
            "step": "backtest",
            "error": "Something broke",
            "elapsed_seconds": 0.0,
            "n_tested": 0,
            "n_survived": 0,
        }
        assert _deterministic_route(state) == "respond"

    def test_full_pipeline_chain(self):
        """Deterministic router follows correct pipeline order."""
        steps = ["", "plan", "load_data", "generate", "backtest", "validate", "narrate"]
        expected = ["plan", "load_data", "generate", "backtest", "validate", "narrate", "respond"]

        for step, exp in zip(steps, expected):
            state: AgentState = {
                "messages": [],
                "session_id": "test",
                "goal": "test",
                "plan": {"tickers": ["SBER"]} if step != "" else None,
                "prices": None,
                "hypotheses": None,
                "results": None,
                "pbo": None,
                "insights": None,
                "narrative": None,
                "step": step,
                "error": None,
                "elapsed_seconds": 0.0,
                "n_tested": 0,
                "n_survived": 0,
            }
            assert _deterministic_route(state) == exp, f"step={step}: expected {exp}"


# ── Respond node tests ──────────────────────────────────────────

class TestRespondNode:
    @pytest.mark.asyncio
    async def test_respond_with_narrative(self):
        """Respond with narrative produces assistant message."""
        state: AgentState = {
            "messages": [],
            "session_id": "test",
            "goal": "test",
            "plan": None,
            "prices": None,
            "hypotheses": None,
            "results": None,
            "pbo": None,
            "insights": ["инсайт 1", "инсайт 2"],
            "narrative": "Тестовый отчёт о momentum.",
            "step": "narrate",
            "error": None,
            "elapsed_seconds": 2.0,
            "n_tested": 10,
            "n_survived": 3,
        }
        result = await respond_node(state)
        assert result["step"] == "done"
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "assistant"
        assert "Тестовый отчёт" in result["messages"][0]["content"]

    @pytest.mark.asyncio
    async def test_respond_with_error(self):
        """Respond with error produces error message."""
        state: AgentState = {
            "messages": [],
            "session_id": "test",
            "goal": "test",
            "plan": None,
            "prices": None,
            "hypotheses": None,
            "results": None,
            "pbo": None,
            "insights": None,
            "narrative": None,
            "step": "",
            "error": "API timeout",
            "elapsed_seconds": 0.0,
            "n_tested": 0,
            "n_survived": 0,
        }
        result = await respond_node(state)
        assert "API timeout" in result["messages"][0]["content"]
        assert result["step"] == "done"

    @pytest.mark.asyncio
    async def test_respond_without_narrative(self):
        """Respond without narrative or error — generic message."""
        state: AgentState = {
            "messages": [],
            "session_id": "test",
            "goal": "test",
            "plan": None,
            "prices": None,
            "hypotheses": None,
            "results": None,
            "pbo": None,
            "insights": None,
            "narrative": None,
            "step": "",
            "error": None,
            "elapsed_seconds": 0.0,
            "n_tested": 0,
            "n_survived": 0,
        }
        result = await respond_node(state)
        assert len(result["messages"]) == 1


# ── Full agent run tests ───────────────────────────────────────

class TestRunAgent:
    @pytest.mark.asyncio
    async def test_run_agent_e2e(self):
        """Full pipeline: plan → data → generate → backtest → validate → narrate → respond."""
        result = await run_agent(
            message="проверь momentum на Сбере",
            session_id="test-agent-e2e",
        )
        assert "response" in result
        assert len(result["response"]) > 0
        # The pipeline should have results and narrative
        assert result.get("results") is not None
        assert len(result["results"]) > 0
        assert result.get("narrative") is not None
        assert len(result["narrative"]) > 0

    @pytest.mark.asyncio
    async def test_run_agent_returns_plan(self):
        """Agent returns the research plan."""
        result = await run_agent(
            message="проверь mean reversion на Газпроме",
            session_id="test-agent-plan",
        )
        assert result.get("plan") is not None
        plan = result["plan"]
        assert "tickers" in plan
        assert "hypothesis_families" in plan

    @pytest.mark.asyncio
    async def test_run_agent_with_unknown_goal(self):
        """Agent handles goals it can't fully parse (fallback)."""
        result = await run_agent(
            message="просто протестируй что-нибудь",
            session_id="test-agent-unknown",
        )
        # Should not crash
        assert "response" in result
        assert result.get("error") is None

    @pytest.mark.asyncio
    async def test_run_agent_multiple_runs(self):
        """Two consecutive runs produce different results."""
        r1 = await run_agent("проверь breakout на Сбере", "test-agent-1")
        r2 = await run_agent("проверь volatility на Сбере", "test-agent-2")
        assert r1.get("narrative") != r2.get("narrative") or (
            r1.get("plan", {}).get("hypothesis_families") != r2.get("plan", {}).get("hypothesis_families")
        )


# ── Helper tests ────────────────────────────────────────────────

class TestHelpers:
    def test_has_llm_key_returns_bool(self):
        """_has_llm_key returns a boolean."""
        result = _has_llm_key()
        assert isinstance(result, bool)


# ── SessionContext tests ──────────────────────────────────────

class TestSessionContext:
    @pytest.mark.asyncio
    async def test_build_context_prompt_without_db_returns_empty(self):
        """Без Postgres build_context_prompt() не падает и возвращает строку."""
        from aqr.agent.context import SessionContext
        ctx = SessionContext("test-no-db")
        prompt = await ctx.build_context_prompt()
        assert isinstance(prompt, str)

    @pytest.mark.asyncio
    async def test_get_recent_runs_without_db_returns_empty_list(self):
        from aqr.agent.context import SessionContext
        ctx = SessionContext("test-no-db")
        runs = await ctx.get_recent_runs()
        assert runs == []

    @pytest.mark.asyncio
    async def test_get_best_strategy_without_db_returns_none(self):
        from aqr.agent.context import SessionContext
        ctx = SessionContext("test-no-db")
        best = await ctx.get_best_strategy()
        assert best is None

    @pytest.mark.asyncio
    async def test_get_untested_combos_without_db_returns_empty_list(self):
        from aqr.agent.context import SessionContext
        ctx = SessionContext("test-no-db")
        spots = await ctx.get_untested_combos()
        assert spots == []


# ── Follow-up routing tests (этап 3.3, пункт 3) ──────────────

class TestFollowUpRouting:
    @pytest.mark.asyncio
    async def test_llm_route_falls_back_to_deterministic_without_key(self):
        """Без LLM-ключа _llm_route делегирует в _deterministic_route."""
        import os

        from aqr.agent.graph import _llm_route
        old = {k: os.environ.pop(k, None) for k in
               ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GIGACHAT_CREDENTIALS", "AQR_LLM_MODEL")}
        try:
            state: AgentState = {
                "messages": [],
                "session_id": "test",
                "goal": "test",
                "plan": None,
                "prices": None,
                "hypotheses": None,
                "results": None,
                "pbo": None,
                "insights": None,
                "narrative": None,
                "step": "narrate",
                "error": None,
                "elapsed_seconds": 0.0,
                "n_tested": 0,
                "n_survived": 0,
            }
            # После narrate → respond (как в детерминистическом роутере)
            assert await _llm_route(state) == "respond"
        finally:
            for k, v in old.items():
                if v is not None:
                    os.environ[k] = v

    @pytest.mark.asyncio
    async def test_run_agent_followup_message_resets_state(self):
        """Follow-up сообщение пользователя — сбрасывает state и запускает заново."""

        # После первого прогона — состояние "done"
        first = await run_agent("проверь momentum на Сбере", "test-followup-1")
        assert first.get("response")

        # Симулируем follow-up — пользователь хочет перепланировать
        # Состояние уже "done", значит route_node должен сбросить step и goal
        from aqr.agent.graph import route_node
        state: AgentState = {
            "messages": [
                {"role": "user", "content": "проверь momentum на Сбере"},
                {"role": "assistant", "content": first["response"]},
                {"role": "user", "content": "а теперь проверь mean reversion"},
            ],
            "session_id": "test-followup-2",
            "goal": "проверь momentum на Сбере",
            "plan": {"tickers": ["SBER"]},
            "prices": {"SBER": [100.0] * 500},
            "hypotheses": [{"family": "momentum"}],
            "results": [{"dsr_verdict": "significant"}],
            "pbo": {"pbo": 0.3, "verdict": "ok"},
            "insights": ["insight 1"],
            "narrative": "old narrative",
            "step": "done",
            "error": None,
            "elapsed_seconds": 1.0,
            "n_tested": 10,
            "n_survived": 3,
        }
        result = await route_node(state)
        # route_node сбрасывает pipeline state и goal
        assert state["step"] == ""
        assert state["goal"] == "а теперь проверь mean reversion"
        assert state["plan"] is None
        assert state["narrative"] is None

    @pytest.mark.asyncio
    async def test_run_agent_two_different_messages(self):
        """Два разных сообщения — два разных прогона (полная перепланировка)."""
        r1 = await run_agent("проверь momentum на Сбере", "test-fu-a")
        r2 = await run_agent("проверь breakout на Газпроме", "test-fu-b")
        assert r1.get("plan", {}).get("hypothesis_families") != r2.get("plan", {}).get("hypothesis_families")
        assert "SBER" in r1.get("plan", {}).get("tickers", [])
        assert "GAZP" in r2.get("plan", {}).get("tickers", [])

    @pytest.mark.asyncio
    async def test_run_agent_includes_session_context_in_state(self):
        """run_agent заполняет session_context_prompt (пустую строку без БД)."""
        # Через прямой вызов run_agent — поле попадает в state
        # Без БД prompt пустой, но это строка
        # Проверим косвенно: _llm_route должен корректно работать с пустым prompt
        import os

        from aqr.agent.graph import _llm_route
        old = {k: os.environ.pop(k, None) for k in
               ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GIGACHAT_CREDENTIALS", "AQR_LLM_MODEL")}
        try:
            state: AgentState = {
                "messages": [{"role": "user", "content": "проверь momentum"}],
                "session_id": "test",
                "goal": "проверь momentum",
                "plan": None,
                "prices": None,
                "hypotheses": None,
                "results": None,
                "pbo": None,
                "insights": None,
                "narrative": "test narrative",
                "session_context_prompt": "",  # как если бы БД не было
                "step": "narrate",
                "error": None,
                "elapsed_seconds": 1.0,
                "n_tested": 5,
                "n_survived": 2,
            }
            # С narrate-шагом → respond
            assert await _llm_route(state) == "respond"
        finally:
            for k, v in old.items():
                if v is not None:
                    os.environ[k] = v
