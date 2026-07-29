"""Orchestrator: runs the 5-agent team in a coordinated flow.

Flow:
    1. Editor plans the goal → ResearchPlan
    2. Browser researches context (parallel with Editor if goal is known)
    3. Analyst runs per-(ticker, family) in parallel via asyncio.gather
    4. Reviewer validates all results
    5. Writer compiles final report
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from aqr.background import schedule

from .analyst import AnalystAgent
from .base import AgentResult
from .browser import BrowserAgent
from .editor import EditorAgent
from .reviewer import ReviewerAgent
from .writer import WriterAgent


@dataclass
class TeamResult:
    """Result of a full team run."""

    ok: bool
    goal: str
    plan: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    results: list[dict[str, Any]] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)
    narrative: str = ""
    insights: list[str] = field(default_factory=list)
    summary: str = ""
    top_results: list[dict[str, Any]] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    n_tested: int = 0
    n_survived: int = 0
    error: str = ""
    agent_errors: list[str] = field(default_factory=list)


async def run_team(
    goal: str,
    session_id: str = "default",
    tickers: list[str] | None = None,
    families: list[str] | None = None,
) -> TeamResult:
    """Run the 5-agent team on a research goal.

    Args:
        goal: research question in Russian (e.g. "проверь momentum на Сбере")
        session_id: session ID for context/credentials
        tickers: optional override (otherwise Editor determines)
        families: optional override (otherwise Editor determines)

    Returns:
        TeamResult with narrative, top_results, validation, etc.
    """
    t0 = time.time()
    errors: list[str] = []

    # Step 1: Editor
    editor = EditorAgent(session_id)
    plan_result = await editor.plan(goal)
    if not plan_result.ok:
        return TeamResult(ok=False, goal=goal, error=plan_result.error, elapsed_seconds=round(time.time() - t0, 2))
    plan = plan_result.data.get("plan", {})

    # Apply overrides
    if tickers is not None:
        plan["tickers"] = tickers
    if families is not None:
        plan["hypothesis_families"] = families

    ticker_list = plan.get("tickers", ["SBER"])
    family_list = plan.get("hypothesis_families", ["momentum"])
    start_date = plan.get("start_date", "2023-01-01")
    end_date = plan.get("end_date", "2024-12-31")
    timeframe = plan.get("timeframe", "D1")

    # Step 2: Browser (context gathering) — runs in parallel with analyst
    browser = BrowserAgent(session_id)
    try:
        browser_task = schedule(browser.research(goal, plan))
    except RuntimeError:
        # Task limit exceeded — run synchronously instead
        browser_task = browser.research(goal, plan)

    # Step 3: Analyst — one per ticker, all families
    analyst = AnalystAgent(session_id)
    analyst_tasks = [
        analyst.analyze(
            ticker=t,
            families=family_list,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
        )
        for t in ticker_list
    ]
    analyst_results = await asyncio.gather(*analyst_tasks, return_exceptions=True)

    # Collect browser context
    context_result: AgentResult | Exception
    try:
        context_result = await browser_task
    except Exception as exc:
        context_result = AgentResult(ok=False, error=f"BrowserAgent: {type(exc).__name__}")
    context_data = context_result.data if isinstance(context_result, AgentResult) and context_result.ok else {}

    # Collect analyst results
    all_results: list[dict] = []
    for i, ar in enumerate(analyst_results):
        if isinstance(ar, Exception):
            errors.append(f"Analyst {ticker_list[i] if i < len(ticker_list) else i}: {ar}")
            continue
        if isinstance(ar, AgentResult) and ar.ok:
            all_results.extend(ar.data.get("results", []))
        elif isinstance(ar, AgentResult):
            errors.append(f"Analyst {ticker_list[i] if i < len(ticker_list) else i}: {ar.error}")

    # Step 4: Reviewer
    reviewer = ReviewerAgent(session_id)
    val_result = await reviewer.validate(all_results, plan)
    validation_data = val_result.data if val_result.ok else {}

    # Step 5: Writer
    writer = WriterAgent(session_id)
    elapsed = round(time.time() - t0, 2)
    write_result = await writer.write(
        goal=goal,
        plan=plan,
        all_results=all_results,
        validation=validation_data,
        elapsed_seconds=elapsed,
    )

    return TeamResult(
        ok=write_result.ok,
        goal=goal,
        plan=plan,
        context=context_data,
        results=all_results,
        validation=validation_data,
        narrative=write_result.data.get("narrative", ""),
        insights=write_result.data.get("insights", []),
        summary=write_result.data.get("summary", ""),
        top_results=write_result.data.get("top_results", []),
        elapsed_seconds=round(time.time() - t0, 2),
        n_tested=validation_data.get("n_tested", len(all_results)),
        n_survived=validation_data.get("n_survived", 0),
        error="; ".join(errors) if errors else "",
        agent_errors=errors,
    )
