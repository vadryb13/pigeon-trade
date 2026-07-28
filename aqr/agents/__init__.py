"""5-agent LangGraph team for parallel quant research.

Roles:
    Editor   — receives goal, creates ResearchPlan, decomposes into subtasks
    Browser  — searches for context (similar hypotheses, ticker info)
    Analyst  — runs VectorBT screener + backtest_one per hypothesis
    Reviewer — PBO / cross-validation / final verdict
    Writer   — compiles final narrative report

Usage:
    from aqr.agents.orchestrator import run_team
    result = await run_team("проверь momentum на Сбере")
    # result.narrative, result.top_results, result.validation
"""

from __future__ import annotations

from .orchestrator import TeamResult, run_team

__all__ = [
    "run_team",
    "TeamResult",
]
