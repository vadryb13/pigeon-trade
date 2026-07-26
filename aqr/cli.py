"""
CLI: `python -m aqr <goal>` — прогон пайплайна в терминале.

Используется:
- людьми: быстро проверить гипотезу без поднятия сервера/UI
- LLM-агентом при разработке: подтвердить что цепочка не сломалась

Пример:
    python -m aqr "проверь momentum на Сбере и Газпроме"
    python -m aqr --json "что работает у металлургов"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

from aqr.logging_config import setup_logging
from aqr.pipeline import ChatPlanner, PipelineExecutor
from aqr.pipeline.events import EventBus

# Configure logging from AQR_LOG_JSON env var (default: human-readable)
setup_logging()


def _fmt_event(ev) -> str:
    """Одна строка события для человека."""
    icon = {
        "planning": "PLAN",
        "data": "DATA",
        "generating": "GEN ",
        "backtesting": "TEST",
        "validating": "VAL ",
        "insight": " *  ",
        "narrating": "TEXT",
        "done": "DONE",
        "error": "ERR ",
    }.get(ev.kind, "    ")
    msg = f"[{icon}] {ev.stage}"
    if ev.message:
        msg += f" — {ev.message}"
    return msg


async def _run(goal: str, as_json: bool, quiet: bool) -> int:
    planner = ChatPlanner()
    plan = planner.plan(goal)

    if not quiet and not as_json:
        print(f"Цель: {plan.goal}")
        print(f"План: {plan.tickers} × {plan.hypothesis_families} × {plan.n_hypotheses} гипотез\n")

    bus = EventBus()
    run_id = bus.new_run()
    executor = PipelineExecutor(bus)

    async def print_events():
        async for ev in bus.subscribe(run_id):
            if not quiet and not as_json:
                print(_fmt_event(ev))
            if ev.kind in ("done", "error"):
                return

    sub_task = asyncio.create_task(print_events())
    await asyncio.sleep(0)  # даём подписаться
    result = await executor.run(run_id, plan)
    try:
        await asyncio.wait_for(sub_task, timeout=2.0)
    except TimeoutError:
        sub_task.cancel()

    if as_json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print("\n" + "─" * 60)
        print("РЕЗУЛЬТАТ")
        print("─" * 60)
        print(f"Проверено гипотез: {result.n_hypotheses_tested}")
        print(f"Прошли Deflated Sharpe: {result.n_survived_dsr}")
        print(f"PBO портфеля: {result.portfolio_pbo:.2f} ({result.portfolio_pbo_verdict})")
        print(f"Время: {result.elapsed_seconds:.1f}s\n")
        if result.top:
            print("Топ-5 по DSR:")
            for r in result.top:
                print(f"  {r.hypothesis.describe():40s}  "
                      f"Sh={r.sharpe:+.2f}  DSR={r.dsr:.2f}  ({r.dsr_verdict})")
        print("\n" + "─" * 60)
        print("НАРРАТИВ")
        print("─" * 60)
        print(result.narrative)
    return 0


def main():
    p = argparse.ArgumentParser(prog="aqr", description="MOEX quant research pipeline")
    p.add_argument("goal", help="Цель на естественном языке")
    p.add_argument("--json", action="store_true", help="Вывести результат как JSON")
    p.add_argument("-q", "--quiet", action="store_true", help="Не показывать промежуточные события")
    args = p.parse_args()
    sys.exit(asyncio.run(_run(args.goal, args.json, args.quiet)))


if __name__ == "__main__":
    main()
