"""WriterAgent: compiles the final narrative report.

Uses the v0.3 `narrate` tool + `extract_insights` tool.
"""

from __future__ import annotations

from typing import Any

from aqr.tools import registry as tool_registry

from .base import AgentResult, BaseAgent


class WriterAgent(BaseAgent):
    """Generates the final narrative from plan + results + validation."""

    name = "writer"

    async def write(
        self,
        goal: str,
        plan: dict[str, Any] | None = None,
        all_results: list[dict[str, Any]] | None = None,
        validation: dict[str, Any] | None = None,
        elapsed_seconds: float = 0.0,
    ) -> AgentResult:
        """Write final report.

        Returns:
            {
                "narrative": str,
                "insights": list[str],
                "summary": str,
                "top_results": list[dict],
            }
        """
        results = all_results or []
        valid = validation or {}

        if not results:
            return self._ok(
                narrative="Не удалось получить результаты исследования.",
                insights=[],
                summary="Нет данных",
                top_results=[],
            )

        # 1) Determine top-5 by DSR
        top = sorted(
            results,
            key=lambda r: r.get("dsr", r.get("sharpe", 0)),
            reverse=True,
        )[:5]

        # 2) Extract deterministic insights
        insights = await self._extract_insights(
            top_results=top,
            n_tested=len(results),
            n_survived=valid.get("n_survived", 0),
            pbo=valid.get("pbo", 0.0),
            pbo_verdict=valid.get("pbo_verdict", ""),
        )

        # 3) Generate narrative via LLM
        narrative = await self._generate_narrative(
            goal=goal,
            plan=plan,
            top_results=top,
            n_tested=len(results),
            n_survived=valid.get("n_survived", 0),
            pbo=valid.get("pbo", 0.0),
            pbo_verdict=valid.get("pbo_verdict", ""),
            elapsed=elapsed_seconds,
        )

        # 4) Build summary line
        summary_parts: list[str] = []
        if top:
            b = top[0]
            summary_parts.append(
                f"Лучшая: {b.get('family', '?')}/{b.get('ticker', '?')} "
                f"DSR={b.get('dsr', 0):.2f}"
            )
        summary_parts.append(f"Проверено: {len(results)}")
        if valid.get("n_survived") is not None:
            summary_parts.append(f"Выжило: {valid['n_survived']}")
        if valid.get("pbo") is not None:
            summary_parts.append(f"PBO={valid['pbo']:.2f}")

        return self._ok(
            narrative=narrative,
            insights=insights,
            summary=" | ".join(summary_parts),
            top_results=top,
        )

    async def _extract_insights(
        self,
        top_results: list[dict],
        n_tested: int,
        n_survived: int,
        pbo: float,
        pbo_verdict: str,
    ) -> list[str]:
        """Run extract_insights tool."""
        tool = tool_registry.get("extract_insights")
        if tool is None:
            return []
        try:
            return await tool.fn(
                top_results=top_results,
                n_tested=n_tested,
                n_survived=n_survived,
                pbo=pbo,
                pbo_verdict=pbo_verdict,
            )
        except Exception:
            self.logger.exception("extract_insights failed")
            return ["Не удалось извлечь инсайты автоматически."]

    async def _generate_narrative(
        self,
        goal: str,
        plan: dict[str, Any] | None,
        top_results: list[dict],
        n_tested: int,
        n_survived: int,
        pbo: float,
        pbo_verdict: str,
        elapsed: float,
    ) -> str:
        """Run narrate tool."""
        tool = tool_registry.get("narrate")
        if tool is None:
            return "Narrate tool not available."

        try:
            return await tool.fn(
                goal=goal,
                tickers=(plan or {}).get("tickers", []),
                families=(plan or {}).get("hypothesis_families", []),
                n_tested=n_tested,
                n_survived=n_survived,
                pbo=pbo,
                pbo_verdict=pbo_verdict,
                top_results=top_results,
                elapsed_seconds=elapsed,
            )
        except Exception:
            self.logger.exception("narrate failed")
            return "Не удалось сгенерировать отчёт."
