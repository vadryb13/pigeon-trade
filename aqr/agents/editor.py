"""EditorAgent: receives a goal, produces a ResearchPlan.

Uses the same `plan_research` tool from v0.3 tool registry.
"""

from __future__ import annotations

from aqr.tools import registry as tool_registry

from .base import AgentResult, BaseAgent


class EditorAgent(BaseAgent):
    """Receives a natural-language goal and returns a structured plan."""

    name = "editor"

    async def plan(self, goal: str) -> AgentResult:
        """Break a goal into a ResearchPlan dictionary.

        Uses the v0.3 `plan_research` tool which calls ChatPlanner + dedup.
        Raises on any failure — strict mode (AGENTS.md invariant 2).
        """
        tool = tool_registry.get("plan_research")
        if tool is None:
            return self._fail(
                "plan_research tool is not registered; "
                "call aqr.tools.register_all() at startup"
            )
        try:
            plan = await tool.fn(goal=goal)
        except Exception as exc:
            return self._fail(f"plan_research failed: {exc}")
        if not plan.get("tickers"):
            return self._fail(
                f"plan_research returned empty tickers for goal={goal!r}"
            )
        return self._ok(plan=plan, fallback=False)
