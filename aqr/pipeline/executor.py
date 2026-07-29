"""
PipelineExecutor — исполняет ResearchPlan шаг за шагом, публикуя события в EventBus.

Шаги:
1. Загрузить данные через T-Invest gRPC (read-through DuckDB cache)
2. Сгенерировать N гипотез (детерминистично из плана)
3. Для каждой: бэктест + Deflated Sharpe + CPCV + PBO по портфелю
4. Ранжировать, оставить топ
5. Готово
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict
from typing import Any

import pandas as pd

from ..types import BacktestResult, PipelineResult
from .events import Event, EventBus
from .hypotheses import HypothesisSpec
from .planner import ResearchPlan

_logger = logging.getLogger(__name__)


class PipelineExecutor:
    """Исполняет план и стримит события."""

    def __init__(self, bus: EventBus):
        self.bus = bus

    async def run(self, run_id: str, plan: ResearchPlan) -> PipelineResult:
        t0 = time.time()

        # Инициализируем реестр инструментов
        from ..tools import registry as tool_registry
        from ..tools.register import register_all
        register_all()

        try:
            await self._emit(run_id, "planning", "План принят",
                             f"{len(plan.tickers)} тикеров, {plan.n_hypotheses} гипотез",
                             {"plan": asdict(plan)})

            # 1. Данные — через инструмент load_prices
            prices = await self._load_data_via_tool(run_id, plan, tool_registry)

            # 2. Гипотезы — через инструмент generate_hypotheses
            await self._emit(run_id, "generating", "Генерирую гипотезы",
                             f"Разбрасываю {plan.n_hypotheses} гипотез по "
                             f"{len(plan.tickers)} тикерам и {len(plan.hypothesis_families)} семействам")
            gen_tool = tool_registry.get("generate_hypotheses")
            specs_raw = await gen_tool.fn(
                tickers=plan.tickers,
                families=plan.hypothesis_families,
                n=plan.n_hypotheses,
            )

            # 3. Бэктест каждой + DSR — через инструмент backtest_one
            bt_tool = tool_registry.get("backtest_one")
            results: list[BacktestResult] = []
            for i, h in enumerate(specs_raw, 1):
                ticker = h["ticker"]
                if ticker not in prices:
                    continue
                validation_cfg = (
                    plan.validation if isinstance(plan.validation, dict) else {}
                )
                bt_raw = await bt_tool.fn(
                    hypothesis=h,
                    prices=prices[ticker].tolist(),
                    n_hypotheses=plan.n_hypotheses,
                    cpcv_splits=int(validation_cfg.get("cpcv_splits", 6)),
                    cpcv_test_splits=int(validation_cfg.get("cpcv_test_splits", 2)),
                    embargo_pct=float(validation_cfg.get("embargo_pct", 0.01)),
                )
                if "error" in bt_raw:
                    continue
                # Конвертируем dict → BacktestResult
                r = self._dict_to_backtest_result(bt_raw)
                results.append(r)
                await self._emit(
                    run_id, "backtesting",
                    f"Бэктест {i}/{len(specs_raw)}",
                    h.get("name", h.get("family", "?")),
                    {
                        "i": i, "n": len(specs_raw),
                        "name": h.get("name", "?"),
                        "sharpe": round(r.sharpe, 2),
                        "dsr_verdict": r.dsr_verdict,
                    },
                )
                await asyncio.sleep(0)

            # 4. Валидация портфеля через PBO — через инструмент validate_portfolio
            await self._emit(run_id, "validating", "PBO по всему портфелю",
                             "Считаю Probability of Backtest Overfitting")
            val_tool = tool_registry.get("validate_portfolio")
            pbo_result = await val_tool.fn(
                results=[r.to_dict() for r in results],
            )

            # 5. Ранжирование
            survived = [r for r in results
                        if r.dsr_verdict in ("significant", "borderline")]
            top = sorted(results, key=lambda r: r.dsr, reverse=True)[:5]

            result = PipelineResult(
                run_id=run_id,
                plan=plan,
                n_hypotheses_tested=len(results),
                n_survived_dsr=len(survived),
                portfolio_pbo=pbo_result["pbo"],
                portfolio_pbo_verdict=pbo_result["verdict"],
                top=top,
                elapsed_seconds=time.time() - t0,
            )

            # Инсайты: сначала детерминистичные — через инструмент extract_insights
            ins_tool = tool_registry.get("extract_insights")
            det_insights = await ins_tool.fn(
                top_results=[r.to_dict() for r in top],
                n_tested=result.n_hypotheses_tested,
                n_survived=result.n_survived_dsr,
                pbo=result.portfolio_pbo,
                pbo_verdict=result.portfolio_pbo_verdict,
            )
            for insight in det_insights:
                await self._emit(run_id, "insight", "Инсайт", insight)

            # LLM-review — через инструмент review_insights
            rev_tool = tool_registry.get("review_insights")
            try:
                extra = await rev_tool.fn(
                    goal=plan.goal,
                    top_results=[r.to_dict() for r in top],
                    deterministic_insights=det_insights,
                    pbo=result.portfolio_pbo,
                    pbo_verdict=result.portfolio_pbo_verdict,
                )
                for obs in extra:
                    await self._emit(run_id, "insight", "Аналитик", obs, {"source": "llm"})
            except Exception:
                _logger.exception("review_insights failed")
                await self._emit(
                    run_id, "warning", "Аналитик",
                    "LLM-review недоступен — показываем только детерминистичные инсайты.",
                    {"source": "llm"},
                )

            # Нарратив — через инструмент narrate
            nar_tool = tool_registry.get("narrate")
            await self._emit(run_id, "narrating", "Пишу резюме", "")
            try:
                result.narrative = await nar_tool.fn(
                    goal=plan.goal,
                    tickers=plan.tickers,
                    families=plan.hypothesis_families,
                    n_tested=result.n_hypotheses_tested,
                    n_survived=result.n_survived_dsr,
                    pbo=result.portfolio_pbo,
                    pbo_verdict=result.portfolio_pbo_verdict,
                    top_results=[r.to_dict() for r in top],
                    elapsed_seconds=result.elapsed_seconds,
                )
            except Exception as e:
                # B12: явно эмитим error — клиент должен видеть, что нарратив
                # не готов. Раньше ошибка тихо записывалась в result.narrative,
                # и pipeline выдавал "done" с мусором в отчёте.
                _logger.exception("narrate failed")
                await self._emit(
                    run_id, "error", "Нарратор",
                    f"narrate failed: {type(e).__name__}",
                    {"exception": type(e).__name__},
                )
                raise

            await self._emit(run_id, "done", "Готово",
                             f"Проверено {len(results)} гипотез, прошло DSR — {len(survived)}",
                             {"result": result.to_dict()})
            return result

        except Exception as e:
            await self._emit(run_id, "error", "Ошибка", type(e).__name__,
                             {"exception": type(e).__name__})
            raise

    # ---------- ХЕЛПЕРЫ ДЛЯ ИНТЕГРАЦИИ С TOOL REGISTRY ----------

    async def _load_data_via_tool(self, run_id: str, plan: ResearchPlan,
                                  tool_registry) -> dict[str, pd.Series]:
        """Загрузка данных через инструмент load_prices с эмиссией событий."""
        load_tool = tool_registry.get("load_prices")
        for t in plan.tickers:
            await self._emit(run_id, "data", f"Загружаю {t}",
                             f"T-Invest: {plan.start_date} → {plan.end_date}")

        prices_raw = await load_tool.fn(
            tickers=plan.tickers,
            start_date=plan.start_date,
            end_date=plan.end_date,
            timeframe=plan.timeframe,
        )

        # Конвертируем list[float] → pd.Series с правильным индексом
        # Используем freq="D" а не "B" — T-Invest D1 возвращает календарные даты
        prices: dict[str, pd.Series] = {}
        n_bars = max((len(v) for v in prices_raw.values()), default=500)
        idx = pd.date_range(plan.start_date, periods=n_bars, freq="D")
        for t, px_list in prices_raw.items():
            s = pd.Series(px_list, index=idx[:len(px_list)], name=t)
            prices[t] = s
            await self._emit(run_id, "data", f"{t}: {len(px_list)} свечей",
                             "OK", {"ticker": t, "n": len(px_list)})
        return prices

    @staticmethod
    def _dict_to_backtest_result(d: dict[str, Any]) -> BacktestResult:
        """Конвертировать словарь из backtest_one в BacktestResult."""
        spec = HypothesisSpec(
            name=d.get("name", "?"),
            family=d.get("family", "?"),
            ticker=d.get("ticker", "?"),
            params=d.get("params", {}),
            fn=lambda x: x,  # не используется после конверсии
        )
        return BacktestResult(
            hypothesis=spec,
            sharpe=d.get("sharpe", 0),
            dsr=d.get("dsr", 0),
            dsr_verdict=d.get("dsr_verdict", "?"),
            cpcv_mean_sharpe=d.get("cpcv_mean_sharpe", 0),
            cpcv_std_sharpe=d.get("cpcv_std_sharpe", 0),
            max_drawdown=d.get("max_drawdown", 0),
            n_trades=d.get("n_trades", 0),
            daily_returns=d.get("daily_returns", []),
        )


    async def _emit(self, run_id: str, kind: str, stage: str,
                    message: str = "", data: dict | None = None):
        await self.bus.publish(Event(
            run_id=run_id, kind=kind, stage=stage, message=message,
            data=data or {},
        ))
        # Структурированный лог: kind → status, stage → tool
        status = "ok" if kind != "error" else "error"
        error_msg = message if status == "error" else None
        from ..logging_config import log_tool_call
        log_tool_call(
            _logger, run_id=run_id, tool=stage, duration_ms=0.0,
            status=status, error=error_msg, kind=kind,
        )
