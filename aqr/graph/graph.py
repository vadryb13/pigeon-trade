"""LangGraph agent: orchestrates pipeline tools in a conversational loop.

Граф:
    plan → load_data → generate → backtest → validate → narrate → respond
    ↑                                                                    │
    └────────────────── follow-up questions ─────────────────────────────┘

Router решает какой узел вызвать на основе текущего состояния.
После ответа пользователю ждёт следующего сообщения — если follow-up,
может перезапустить часть пайплайна.

Caveat: follow-up routing через `_llm_route` работает только при
наличии `AQR_LLM_MODEL` + API-ключа. Без LLM граф после первого
`respond` завершается через END — никакого follow-up не происходит.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from ..tools import registry as tool_registry
from ..tools.register import register_all
from .context import SessionContext

# Гарантируем что инструменты зарегистрированы
register_all()


def _msg_role(msg) -> str:
    """Извлечь роль из сообщения (dict или LangChain-объект)."""
    if hasattr(msg, "content"):
        type_name = type(msg).__name__
        if "Human" in type_name:
            return "user"
        if "AI" in type_name:
            return "assistant"
        if "Tool" in type_name:
            return "tool"
    if isinstance(msg, dict):
        return msg.get("role", "user")
    return "user"


def _msg_content(msg) -> str:
    """Извлечь текст из сообщения (dict или LangChain-объект)."""
    if hasattr(msg, "content"):
        content = msg.content
        if isinstance(content, list):
            return " ".join(
                p.get("text", "") if isinstance(p, dict) else str(p)
                for p in content
            )
        return str(content) if content else ""
    if isinstance(msg, dict):
        return msg.get("content", "")
    return ""

# ── State ───────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    """Состояние агента — передаётся между узлами графа."""

    messages: Annotated[list[dict[str, Any]], add_messages]
    session_id: str
    # Pipeline state
    goal: str
    plan: dict[str, Any] | None
    prices: dict[str, list[float]] | None
    hypotheses: list[dict[str, Any]] | None
    results: list[dict[str, Any]] | None
    pbo: dict[str, Any] | None
    insights: list[str] | None
    narrative: str | None
    # Session-augmented prompt (для LLM-узлов)
    session_context_prompt: str
    # Control
    step: str  # plan | load | generate | backtest | validate | narrate | done
    error: str | None
    # Metadata
    elapsed_seconds: float
    n_tested: int
    n_survived: int


# ── System prompt ───────────────────────────────────────────────

ROUTER_SYSTEM = """Ты — ассистент quant-исследователя. Твоя задача — провести пользователя
через исследование: понять цель, спланировать, загрузить данные, проверить гипотезы,
провалидировать и дать отчёт.

Ты получаешь состояние исследования (что уже сделано) и сообщение пользователя.
Твоя задача — вернуть JSON с одним полем "action":

{{
  "action": "<следующий шаг>"
}}

Где <следующий шаг> — одно из:
- "plan" — цель ещё не разобрана в план
- "load_data" — план есть, данные не загружены
- "generate" — данные есть, гипотезы не сгенерированы
- "backtest" — гипотезы есть, бэктест не проведён
- "validate" — бэктест проведён, валидация не сделана
- "narrate" — всё готово, нужно сформировать отчёт
- "respond" — ответить пользователю
- "done" — завершить диалог

Правила:
1. Если это первое сообщение — всегда "plan".
2. Если пользователь просит повторить или уточнить (например "а теперь проверь mean reversion") —
   верни "plan" чтобы перепланировать.
3. Если пользователь просит убрать/добавить семейство гипотез — верни "generate".
4. Если уже есть narrative и пользователь задаёт вопрос о результате — ответь "respond".
5. Если произошла ошибка — ответь "respond" с объяснением ошибки.
6. Не придумывай действий, которых нет в списке.
"""

RESPOND_SYSTEM = """Ты — ассистент quant-исследователя. Ты помогаешь анализировать
результаты проверки торговых гипотез. Отвечай по-русски, кратко, по делу.

У тебя есть:
- Результаты бэктеста (топ гипотез с Sharpe, DSR, просадкой)
- Вердикт PBO (вероятность переобучения портфеля)
- Нарратив (автоматически сгенерированный отчёт)

Твоя задача — ответить на вопрос пользователя, используя эти данные.
Если пользователь спрашивает о метриках — объясни их значения.
Если просит рекомендацию — предложи на основе DSR и PBO.
Не выдумывай данных, которых нет в результатах."""


# ── Nodes ───────────────────────────────────────────────────────

async def plan_node(state: AgentState) -> dict[str, Any]:
    """Разобрать цель в ResearchPlan."""
    goal = state.get("goal", "")
    if not goal and state.get("messages"):
        goal = _msg_content(state["messages"][-1])

    tool = tool_registry.get("plan_research")
    plan = await tool.fn(goal=goal)

    return {
        "plan": plan,
        "goal": goal,
        "step": "plan",
        "tickers": plan.get("tickers", []),
    }


async def load_data_node(state: AgentState) -> dict[str, Any]:
    """Загрузить цены для тикеров из плана."""
    plan = state["plan"]
    tool = tool_registry.get("load_prices")
    prices = await tool.fn(
        tickers=plan.get("tickers", []),
        start_date=plan.get("start_date", "2023-01-01"),
        end_date=plan.get("end_date", "2024-12-31"),
        timeframe=plan.get("timeframe", "D1"),
    )
    return {"prices": prices, "step": "load_data"}


async def generate_node(state: AgentState) -> dict[str, Any]:
    """Сгенерировать гипотезы по плану."""
    plan = state["plan"]
    tool = tool_registry.get("generate_hypotheses")
    hypotheses = await tool.fn(
        tickers=plan.get("tickers", []),
        families=plan.get("hypothesis_families", []),
        n=plan.get("n_hypotheses", 20),
    )
    return {"hypotheses": hypotheses, "step": "generate"}


async def backtest_node(state: AgentState) -> dict[str, Any]:
    """Пробэктестировать все гипотезы."""
    hypotheses = state["hypotheses"]
    prices = state["prices"]
    plan = state["plan"]
    n_hypotheses = plan.get("n_hypotheses", 20)
    # CPCV/validation config — план может переопределить дефолты tools
    validation = plan.get("validation") or {}

    tool = tool_registry.get("backtest_one")
    results = []
    for h in hypotheses:
        ticker = h["ticker"]
        if ticker not in prices:
            continue
        r = await tool.fn(
            hypothesis=h,
            prices=prices[ticker],
            n_hypotheses=n_hypotheses,
            cpcv_splits=int(validation.get("cpcv_splits", 6)),
            cpcv_test_splits=int(validation.get("cpcv_test_splits", 2)),
            embargo_pct=float(validation.get("embargo_pct", 0.01)),
        )
        if "error" not in r:
            results.append(r)
        await asyncio.sleep(0)  # даём event loop дышать

    return {"results": results, "step": "backtest", "n_tested": len(results)}


async def validate_node(state: AgentState) -> dict[str, Any]:
    """PBO-валидация портфеля результатов."""
    results = state["results"]
    tool = tool_registry.get("validate_portfolio")
    pbo = await tool.fn(results=results)
    return {"pbo": pbo, "step": "validate"}


async def narrate_node(state: AgentState) -> dict[str, Any]:
    """Сформировать отчёт: инсайты + нарратив."""
    results = state["results"]
    plan = state["plan"]
    pbo = state.get("pbo", {})

    # Топ-5 по DSR
    top = sorted(results, key=lambda r: r.get("dsr", 0), reverse=True)[:5]
    survived = sum(
        1 for r in results
        if r.get("dsr_verdict") in ("significant", "borderline")
    )

    # Инсайты
    ins_tool = tool_registry.get("extract_insights")
    insights = await ins_tool.fn(
        top_results=top,
        n_tested=len(results),
        n_survived=survived,
        pbo=pbo.get("pbo", 0),
        pbo_verdict=pbo.get("verdict", ""),
    )

    # LLM-review (может вернуть [])
    try:
        rev_tool = tool_registry.get("review_insights")
        extra = await rev_tool.fn(
            goal=state.get("goal", ""),
            top_results=top,
            deterministic_insights=insights,
            pbo=pbo.get("pbo", 0),
            pbo_verdict=pbo.get("verdict", ""),
        )
        insights.extend(extra)
    except Exception:
        # B20: LLM-review — дополнительная фича. Логируем, но не валим граф.
        logging.getLogger(__name__).exception("review_insights failed")

    # Нарратив
    nar_tool = tool_registry.get("narrate")
    narrative = await nar_tool.fn(
        goal=state.get("goal", ""),
        tickers=plan.get("tickers", []),
        families=plan.get("hypothesis_families", []),
        n_tested=len(results),
        n_survived=survived,
        pbo=pbo.get("pbo", 0),
        pbo_verdict=pbo.get("verdict", ""),
        top_results=top,
        elapsed_seconds=state.get("elapsed_seconds", 0),
    )

    return {
        "insights": insights,
        "narrative": narrative,
        "step": "narrate",
        "n_survived": survived,
    }


async def respond_node(state: AgentState) -> dict[str, Any]:
    """Ответить пользователю на основе результатов."""
    narrative = state.get("narrative", "")
    error = state.get("error")

    if error:
        response = f"При выполнении возникла ошибка: {error}"
    elif narrative:
        insights = state.get("insights", [])
        insight_text = "\n".join(f"• {i}" for i in insights) if insights else ""
        response = f"{narrative}\n\nКлючевые наблюдения:\n{insight_text}" if insight_text else narrative
    else:
        response = "Исследование завершено. Результаты недоступны."

    return {
        "messages": [{"role": "assistant", "content": response}],
        "step": "done",
    }


# ── Router ──────────────────────────────────────────────────────

def _deterministic_route(state: AgentState) -> str:
    """Детерминистический роутер: проверяет состояние и возвращает следующий узел.

    Порядок: plan → load_data → generate → backtest → validate → narrate → respond
    """
    step = state.get("step", "")

    if state.get("error"):
        return "respond"

    # Цепочка по шагам
    pipeline_order = {
        "": "plan",
        "plan": "load_data",
        "load_data": "generate",
        "generate": "backtest",
        "backtest": "validate",
        "validate": "narrate",
        "narrate": "respond",
    }

    if step in pipeline_order:
        return pipeline_order[step]

    # Если step == "done" или неизвестный — завершаем
    if step == "done":
        return END

    return "respond"


async def _llm_route(state: AgentState) -> str:
    """LLM-роутер: анализирует состояние и сообщение пользователя.

    Используется после первого ответа для обработки follow-up вопросов.
    Возвращает узел для следующего шага.
    """
    # Если пайплайн завершён — не вызываем LLM, сразу END
    if state.get("step") == "done":
        return END

    # Если нет LLM — детерминистический роутер.
    # Проверяем и env, и per-session credentials (ContextVar) для WS-режима.
    if not _has_llm_key():
        return _deterministic_route(state)
    model = os.environ.get("AQR_LLM_MODEL")
    if not model:
        from .context import current_credentials
        creds = current_credentials()
        model = creds.llm_model if creds else ""
    if not model:
        return _deterministic_route(state)

    messages = state.get("messages", [])
    if not messages:
        return _deterministic_route(state)

    # Строим контекст для LLM
    context_parts = [
        f"Цель исследования: {state.get('goal', 'не задана')}",
        f"Текущий шаг: {state.get('step', 'начало')}",
    ]
    if state.get("plan"):
        p = state["plan"]
        context_parts.append(
            f"План: тикеры={p.get('tickers')}, семейства={p.get('hypothesis_families')}, "
            f"гипотез={p.get('n_hypotheses')}"
        )
    if state.get("narrative"):
        context_parts.append(f"Отчёт уже сгенерирован ({len(state['narrative'])} символов).")
    if state.get("error"):
        context_parts.append(f"Ошибка: {state['error']}")

    context = "\n".join(context_parts)
    user_msg = _msg_content(messages[-1])

    # Смешиваем контекст сессии в системный промпт, если есть
    session_prompt = state.get("session_context_prompt", "")
    system_content = ROUTER_SYSTEM
    if session_prompt:
        system_content = ROUTER_SYSTEM + "\n\nКонтекст сессии:\n" + session_prompt

    try:
        import litellm
        resp = await litellm.acompletion(
            model=model,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"Состояние:\n{context}\n\nСообщение пользователя: {user_msg}"},
            ],
            response_format={"type": "json_object"},
        )
        import json
        data = json.loads(resp.choices[0].message.content)
        action = data.get("action", "respond")

        valid_actions = {"plan", "load_data", "generate", "backtest", "validate", "narrate", "respond"}
        if action in valid_actions:
            if action == "respond":
                return "respond"
            if action == "done":
                return END
            return action
    except Exception:
        logging.getLogger(__name__).exception(
            "LLM routing failed, falling back to deterministic"
        )
    return _deterministic_route(state)


def _has_llm_key() -> bool:
    """Check if any LLM credentials are available (env or per-session ContextVar)."""
    keys = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GIGACHAT_CREDENTIALS", "DEEPSEEK_API_KEY")
    if any(os.environ.get(k) for k in keys):
        return True
    try:
        from .context import current_credentials
        creds = current_credentials()
        if creds and creds.llm_api_key:
            return True
    except Exception:
        pass
    return False


async def route_node(state: AgentState) -> dict[str, Any]:
    """Точка входа: принимает сообщение пользователя и решает что делать."""
    messages = state.get("messages", [])
    if messages:
        last_msg = messages[-1]
        if _msg_role(last_msg) == "user":
            state["goal"] = _msg_content(last_msg)
            # Сброс состояния для нового исследования
            if state.get("step") == "done":
                state.update({
                    "plan": None, "prices": None, "hypotheses": None,
                    "results": None, "pbo": None, "insights": None,
                    "narrative": None, "step": "", "error": None,
                })
                return {"step": ""}

    # Если состояние "done" и пришло новое сообщение — начинаем заново
    # Возвращаем шаг явно — мутация state dict`а не влияет на граф
    if state.get("step") == "done":
        return {"step": ""}

    return {}


# ── Build graph ─────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Построить LangGraph-граф агента."""
    graph = StateGraph(AgentState)

    # Добавляем узлы
    graph.add_node("route", route_node)
    graph.add_node("plan", plan_node)
    graph.add_node("load_data", load_data_node)
    graph.add_node("generate", generate_node)
    graph.add_node("backtest", backtest_node)
    graph.add_node("validate", validate_node)
    graph.add_node("narrate", narrate_node)
    graph.add_node("respond", respond_node)

    # Точка входа: всегда начинаем с route
    graph.set_entry_point("route")

    # После route — определяем куда идти
    graph.add_conditional_edges(
        "route",
        _deterministic_route,
        {
            "plan": "plan",
            "load_data": "load_data",
            "generate": "generate",
            "backtest": "backtest",
            "validate": "validate",
            "narrate": "narrate",
            "respond": "respond",
            END: END,
        },
    )

    # Линейная цепочка
    graph.add_edge("plan", "load_data")
    graph.add_edge("load_data", "generate")
    graph.add_edge("generate", "backtest")
    graph.add_edge("backtest", "validate")
    graph.add_edge("validate", "narrate")
    graph.add_edge("narrate", "respond")

    # После respond — ждём следующего сообщения
    graph.add_conditional_edges(
        "respond",
        _llm_route,
        {
            "plan": "plan",
            "load_data": "load_data",
            "generate": "generate",
            "backtest": "backtest",
            "validate": "validate",
            "narrate": "narrate",
            "respond": "respond",
            END: END,
        },
    )

    return graph


# Глобальный скомпилированный граф
_agent_graph = None


def get_agent():
    """Получить скомпилированный граф агента (синглтон)."""
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = build_graph().compile()
    return _agent_graph


# ── High-level API ──────────────────────────────────────────────

async def run_agent(
    message: str,
    session_id: str = "default",
) -> dict[str, Any]:
    """Запустить агента с сообщением пользователя.

    Returns:
        {"response": str, "narrative": str | None, "results": list | None, ...}
    """
    t0 = time.time()
    agent = get_agent()

    # Собираем контекст сессии (история, лучшая стратегия, белые пятна) —
    # подмешивается в LLM-промпт в `_llm_route`. Без LLM этот контекст никем
    # не используется, поэтому пропускаем 3 DB-запроса (PERF-5).
    session_context_prompt = ""
    if _has_llm_key():
        try:
            session_context_prompt = await SessionContext(
                session_id,
            ).build_context_prompt()
        except Exception:
            # Не критично — без контекста агент всё равно работает
            session_context_prompt = ""

    initial_state: AgentState = {
        "messages": [{"role": "user", "content": message}],
        "session_id": session_id,
        "goal": message,
        "plan": None,
        "prices": None,
        "hypotheses": None,
        "results": None,
        "pbo": None,
        "insights": None,
        "narrative": None,
        "session_context_prompt": session_context_prompt,
        "step": "",
        "error": None,
        "elapsed_seconds": 0.0,
        "n_tested": 0,
        "n_survived": 0,
    }

    try:
        final_state = await agent.ainvoke(initial_state)
    except Exception as e:
        logging.getLogger(__name__).exception("run_agent failed")
        return {
            "response": "Ошибка при выполнении",
            "narrative": None,
            "results": None,
            "error": type(e).__name__,
        }

    messages = final_state.get("messages", [])
    response = ""
    for m in reversed(messages):
        if _msg_role(m) == "assistant":
            response = _msg_content(m)
            break

    return {
        "response": response,
        "narrative": final_state.get("narrative"),
        "results": final_state.get("results"),
        "plan": final_state.get("plan"),
        "pbo": final_state.get("pbo"),
        "insights": final_state.get("insights"),
        "elapsed_seconds": round(time.time() - t0, 2),
    }
