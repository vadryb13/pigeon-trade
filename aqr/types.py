"""Shared dataclasses, используемые несколькими слоями.

`BacktestResult` и `PipelineResult` лежат здесь, а не в
`aqr/pipeline/executor.py`, чтобы разорвать потенциальный цикл импортов:
`aqr/pipeline/executor.py` импортирует `aqr/tools/register.py`, а
`aqr/tools/core.py` исторически импортировал `BacktestResult` из executor.
Если tools начнёт импортировать что-то, что тянет executor — получим цикл.

Сейчас здесь:
- `BacktestResult` — результат одного бэктеста.
- `PipelineResult` — финальный результат прогона.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

# Размерность вектора эмбеддинга. Должна совпадать с Vector(N) в models.py.
# Менять синхронно: здесь, в models.py (Vector(N)), и через ALTER TABLE.
EMBEDDING_DIM = 768

if TYPE_CHECKING:
    # Аннотации типов для IDE/static-checkers — runtime не нужен импорт,
    # потому что dataclasses используют только type hints и forward-refs.
    from .pipeline.hypotheses import HypothesisSpec
    from .pipeline.planner import ResearchPlan


@dataclass
class BacktestResult:
    hypothesis: HypothesisSpec
    sharpe: float
    dsr: float
    dsr_verdict: str
    cpcv_mean_sharpe: float
    cpcv_std_sharpe: float
    max_drawdown: float
    n_trades: int
    daily_returns: list[float]  # для PBO

    def to_dict(self) -> dict[str, Any]:
        h = self.hypothesis
        return {
            "name": h.name,
            "family": h.family,
            "ticker": h.ticker,
            "params": h.params,
            "sharpe": round(self.sharpe, 3),
            "dsr": round(self.dsr, 3),
            "dsr_verdict": self.dsr_verdict,
            "cpcv_mean_sharpe": round(self.cpcv_mean_sharpe, 3),
            "cpcv_std_sharpe": round(self.cpcv_std_sharpe, 3),
            "max_drawdown": round(self.max_drawdown, 3),
            "n_trades": self.n_trades,
            "daily_returns": self.daily_returns,
        }


@dataclass
class PipelineResult:
    run_id: str
    plan: ResearchPlan
    n_hypotheses_tested: int
    n_survived_dsr: int
    portfolio_pbo: float
    portfolio_pbo_verdict: str
    top: list[BacktestResult] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    narrative: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "plan": asdict(self.plan),
            "n_hypotheses_tested": self.n_hypotheses_tested,
            "n_survived_dsr": self.n_survived_dsr,
            "portfolio_pbo": round(self.portfolio_pbo, 3),
            "portfolio_pbo_verdict": self.portfolio_pbo_verdict,
            "top": [r.to_dict() for r in self.top],
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "narrative": self.narrative,
        }


__all__ = ["BacktestResult", "PipelineResult"]
