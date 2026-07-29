"""ReviewerAgent: validates all backtest results.

Runs:
  1) PBO on the portfolio of daily_returns
  2) Aggregate statistics (survival rate, best/worst)
  3) Recommendations for the writer
"""

from __future__ import annotations

from typing import Any

from aqr.tools import registry as tool_registry

from .base import AgentResult, BaseAgent


class ReviewerAgent(BaseAgent):
    """Reviews all analyst results: PBO, aggregate metrics, verdict."""

    name = "reviewer"

    async def validate(
        self,
        all_results: list[dict[str, Any]],
        plan: dict[str, Any] | None = None,
    ) -> AgentResult:
        """Run validation on the full result set.

        Returns:
            {
                "pbo": float,
                "pbo_verdict": str,
                "n_tested": int,
                "n_survived": int,
                "survival_rate": float,
                "best_result": dict | None,
                "aggregate": dict,
                "recommendations": list[str],
            }
        """
        if not all_results:
            return self._ok(
                pbo=0.0,
                pbo_verdict="no_data",
                n_tested=0,
                n_survived=0,
                survival_rate=0.0,
                best_result=None,
                aggregate={},
                recommendations=["Нет результатов для проверки"],
            )

        # 1) PBO via validate_portfolio tool
        pbo_data = await self._compute_pbo(all_results)

        # 2) Aggregate statistics
        n_tested = len(all_results)
        n_survived = sum(
            1 for r in all_results
            if r.get("dsr_verdict") in ("significant", "borderline")
        )
        best = max(
            all_results,
            key=lambda r: r.get("dsr", r.get("sharpe", 0)),
            default=None,
        )

        sharpe_vals = [r.get("sharpe", 0) for r in all_results]
        dsr_vals = [r.get("dsr", 0) for r in all_results]

        # 3) Recommendations
        recs: list[str] = []
        if n_survived == 0 and n_tested > 0:
            recs.append("Ни одна гипотеза не прошла Deflated Sharpe — попробуй другие параметры")
        if pbo_data.get("pbo", 0) > 0.5:
            recs.append(f"PBO={pbo_data['pbo']:.2f} — высокий риск переобучения")
        if best:
            recs.append(
                f"Лучшая: {best.get('family', '?')}/{best.get('ticker', '?')} "
                f"DSR={best.get('dsr', 0):.2f}, Sharpe={best.get('sharpe', 0):.2f}"
            )

        return self._ok(
            pbo=pbo_data.get("pbo", 0.0),
            pbo_verdict=pbo_data.get("pbo_verdict", ""),
            n_tested=n_tested,
            n_survived=n_survived,
            survival_rate=n_survived / n_tested if n_tested > 0 else 0.0,
            best_result=best,
            aggregate={
                "mean_sharpe": sum(sharpe_vals) / len(sharpe_vals) if sharpe_vals else 0.0,
                "max_sharpe": max(sharpe_vals) if sharpe_vals else 0.0,
                "mean_dsr": sum(dsr_vals) / len(dsr_vals) if dsr_vals else 0.0,
                "max_dsr": max(dsr_vals) if dsr_vals else 0.0,
            },
            recommendations=recs,
        )

    async def _compute_pbo(self, results: list[dict]) -> dict[str, Any]:
        """Delegate PBO computation to validate_portfolio tool."""
        tool = tool_registry.get("validate_portfolio")
        if tool is None:
            return {"pbo": 0.0, "pbo_verdict": "tool_missing"}
        try:
            return await tool.fn(results=results)
        except Exception as exc:
            self.logger.exception("PBO computation failed")
            return {"pbo": 0.0, "pbo_verdict": f"error: {type(exc).__name__}"}
