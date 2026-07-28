"""Session context for ChatAgent: recent runs, best strategy, untested combos.

Подмешивается в системный промпт LLM-роутера и narrate-узла, чтобы агент
помнил историю сессии и предлагал непроверенные комбинации.

Также: per-session credentials через ContextVar (`current_credentials`).
Устанавливается в `aqr.chat.ws._run_agent_for_session` на время одного
прогонов графа. Читается `current_credentials()` из planner/narrator/
reviewer/embedder/tinvest — без credentials LLM/Invest не работают.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar

logger = logging.getLogger(__name__)

from typing import Any

from ..db import _async_session_factory
from ..registry import DecryptedSettings, RegistryStore

# ── Per-session credentials ─────────────────────────────────────

_active_credentials: ContextVar[DecryptedSettings | None] = ContextVar(
    "aqr_active_credentials", default=None
)


def set_credentials(settings: DecryptedSettings | None) -> Any:
    """Установить credentials в текущий контекст. Возвращает token для reset."""
    return _active_credentials.set(settings)


def reset_credentials(token: Any) -> None:
    """Восстановить предыдущее значение credentials."""
    _active_credentials.reset(token)


def current_credentials() -> DecryptedSettings | None:
    """Credentials текущей сессии или None если не установлены.

    Инструменты (plan_research, load_prices, narrate, Embedder и т.д.)
    читают credentials через эту функцию. Если None — runtime-ошибка
    «settings not configured» (см. AGENTS.md инвариант 2).
    """
    return _active_credentials.get()


class SessionContext:
    """Контекст сессии, подмешиваемый в системный промпт агента."""

    def __init__(self, session_id: str = "default") -> None:
        self.session_id = session_id

    async def get_recent_runs(self, limit: int = 5) -> list[dict[str, Any]]:
        """Последние прогоны в сессии. Пустой список, если БД недоступна."""
        try:
            async with _async_session_factory() as db:
                store = RegistryStore(db)
                runs = await store.list_runs_by_session(self.session_id, limit=limit)
                return [
                    {
                        "id": str(r.id),
                        "goal": r.goal,
                        "status": r.status,
                        "metrics": r.summary_metrics,
                    }
                    for r in runs
                ]
        except Exception:
            logger.exception("get_recent_runs failed for session=%s", self.session_id)
            return []

    async def get_best_strategy(self) -> dict[str, Any] | None:
        """Лучшая гипотеза по DSR среди всех прогонов сессии. None, если БД недоступна."""
        try:
            async with _async_session_factory() as db:
                store = RegistryStore(db)
                runs = await store.list_runs_by_session(self.session_id, limit=20)
                if not runs:
                    return None
                # B11: один батч-запрос вместо N+1.
                goals_map = {r.id: r.goal for r in runs}
                by_run = await store.list_hypotheses_by_runs([r.id for r in runs])
                best_hyp = None
                best_dsr = -999.0
                for run_id, hyps in by_run.items():
                    for h in hyps:
                        if h.dsr is not None and h.dsr > best_dsr:
                            best_dsr = h.dsr
                            best_hyp = {
                                "run_id": str(run_id),
                                "goal": goals_map.get(run_id, ""),
                                "family": h.family,
                                "ticker": h.ticker,
                                "params": h.config_json,
                                "dsr": h.dsr,
                                "sharpe": h.sharpe,
                                "is_valid": h.is_valid,
                            }
                return best_hyp
        except Exception:
            logger.exception("get_best_strategy failed for session=%s", self.session_id)
            return None

    async def get_untested_combos(self) -> list[dict[str, Any]]:
        """Семейства × тикеры, которые ещё не проверялись в этой сессии.

        Возвращает до 5 предложений что можно проверить. Пустой список без БД.
        """
        try:
            async with _async_session_factory() as db:
                store = RegistryStore(db)
                runs = await store.list_runs_by_session(self.session_id, limit=50)
                if not runs:
                    return []

                all_families = {"momentum", "mean_reversion", "breakout", "volatility"}
                tested: set[tuple[str, str]] = set()
                tested_tickers: set[str] = set()

                # B11: один батч-запрос вместо N+1.
                by_run = await store.list_hypotheses_by_runs([r.id for r in runs])
                for hyps in by_run.values():
                    for h in hyps:
                        tested.add((h.family, h.ticker))
                        tested_tickers.add(h.ticker)

                suggestions = []
                for ticker in list(tested_tickers)[:5]:
                    for fam in all_families:
                        if (fam, ticker) not in tested:
                            suggestions.append({"family": fam, "ticker": ticker})
                            if len(suggestions) >= 5:
                                return suggestions

                all_tickers = {"SBER", "GAZP", "LKOH", "GMKN", "ROSN", "TATN", "CHMF", "NLMK"}
                for ticker in all_tickers - tested_tickers:
                    suggestions.append({"family": "momentum", "ticker": ticker})
                    if len(suggestions) >= 5:
                        return suggestions

                return suggestions
        except Exception:
            logger.exception("get_untested_combos failed for session=%s", self.session_id)
            return []

    async def build_context_prompt(self) -> str:
        """Собрать контекст сессии в строку для системного промпта.

        Возвращает пустую строку, если БД недоступна — это позволяет
        агенту работать в тестах без Postgres.
        """
        parts: list[str] = []

        try:
            recent = await self.get_recent_runs(3)
            if recent:
                parts.append("Последние прогоны в этой сессии:")
                for r in recent:
                    metrics = r.get("metrics") or {}
                    parts.append(
                        f"  - [{r['id'][:8]}...] {r['goal']} "
                        f"(статус: {r['status']}, "
                        f"проверено: {metrics.get('n_tested', '?')}, "
                        f"выжило DSR: {metrics.get('n_survived_dsr', '?')}, "
                        f"PBO: {metrics.get('portfolio_pbo', '?')})"
                    )

            best = await self.get_best_strategy()
            if best:
                parts.append(
                    f"\nЛучшая стратегия в сессии: "
                    f"{best['family']}/{best['ticker']} "
                    f"(DSR={best['dsr']:.2f}, Sharpe={best['sharpe']:.2f}, "
                    f"params={best['params']})"
                )

            spots = await self.get_untested_combos()
            if spots:
                parts.append("\nНепроверенные комбинации (белые пятна):")
                for s in spots[:5]:
                    parts.append(f"  - {s['family']} на {s['ticker']}")
        except Exception:
            logger.exception("build_context_prompt failed for session=%s", self.session_id)
            return ""

        return "\n".join(parts)
