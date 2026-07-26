"""End-to-end тест сквозного пайплайна без LLM и без сети."""
from __future__ import annotations

import asyncio
import os

# Форсируем synthetic-путь: отключаем LLM и MOEX
os.environ.pop("ANTHROPIC_API_KEY", None)
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("AQR_LLM_MODEL", None)


def test_planner_fallback_extracts_ticker_and_family():
    from aqr.pipeline import ChatPlanner
    p = ChatPlanner().plan("Проверь momentum на Сбере и Газпроме на дневках")
    assert "SBER" in p.tickers
    assert "GAZP" in p.tickers
    assert "momentum" in p.hypothesis_families
    assert p.timeframe == "D1"


def test_planner_blue_chips():
    from aqr.pipeline import ChatPlanner
    p = ChatPlanner().plan("Найди устойчивые стратегии на голубых фишках")
    assert any(t in p.tickers for t in ("SBER", "GAZP", "LKOH"))


def test_planner_metallurgy():
    from aqr.pipeline import ChatPlanner
    p = ChatPlanner().plan("что работает у металлургов")
    assert any(t in p.tickers for t in ("CHMF", "NLMK", "MAGN", "PLZL"))


def test_pipeline_synthetic_end_to_end(monkeypatch):
    """
    Полный прогон без сети: планировщик → executor → narrator.
    Проверяем, что события идут, результат заполнен, нарратив непустой.
    """
    # Форсируем synthetic-путь: сломаем MOEXAdapter
    from aqr.data import moex as moex_mod
    from aqr.pipeline import ChatPlanner, PipelineExecutor
    from aqr.pipeline.events import EventBus

    class _BrokenAdapter:
        def __init__(self, *a, **kw): pass
        def candles(self, *a, **kw):
            raise RuntimeError("no network in test")

    monkeypatch.setattr(moex_mod, "MOEXAdapter", _BrokenAdapter)

    plan = ChatPlanner().plan("проверь momentum и mean_reversion на Сбере")
    plan.n_hypotheses = 8

    bus = EventBus()
    run_id = bus.new_run()

    events_collected = []

    async def collector():
        # Стартуем подписку до запуска executor
        async for ev in bus.subscribe(run_id):
            events_collected.append(ev)

    async def go():
        # Собираем задачи параллельно
        sub = asyncio.create_task(collector())
        await asyncio.sleep(0.01)  # даём подписаться
        result = await PipelineExecutor(bus).run(run_id, plan)
        # Ждём, чтобы подписчик успел получить done
        try:
            await asyncio.wait_for(sub, timeout=2.0)
        except TimeoutError:
            sub.cancel()
        return result

    result = asyncio.run(go())

    # Проверки результата
    assert result.n_hypotheses_tested > 0
    assert result.top, "должен быть непустой топ"
    assert result.narrative, "нарратив должен быть заполнен fallback-шаблоном"
    assert result.plan.tickers == ["SBER"]

    # Проверки событий
    kinds = [e.kind for e in events_collected]
    assert "planning" in kinds
    assert "generating" in kinds
    assert "backtesting" in kinds
    assert "validating" in kinds
    assert "insight" in kinds
    assert "done" in kinds
    # порядок: planning раньше done
    assert kinds.index("planning") < kinds.index("done")


def test_narrator_fallback_contains_key_facts():
    from aqr.data import moex as moex_mod
    from aqr.pipeline import ChatPlanner, PipelineExecutor
    from aqr.pipeline.events import EventBus

    class _BrokenAdapter:
        def __init__(self, *a, **kw): pass
        def candles(self, *a, **kw):
            raise RuntimeError("offline")

    import unittest.mock
    with unittest.mock.patch.object(moex_mod, "MOEXAdapter", _BrokenAdapter):
        plan = ChatPlanner().plan("проверь momentum на Сбере")
        plan.n_hypotheses = 6
        bus = EventBus()
        rid = bus.new_run()
        result = asyncio.run(PipelineExecutor(bus).run(rid, plan))

    text = result.narrative
    assert "momentum" in text.lower() or "SBER" in text
    assert "Sharpe" in text or "sharpe" in text.lower()
    # 3-6 абзацев
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    assert 2 <= len(paragraphs) <= 8
