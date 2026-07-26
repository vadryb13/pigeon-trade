"""Core pipeline tools: plan, load, generate, backtest, validate, insights, narrate.

Каждый инструмент — асинхронная функция с явной сигнатурой.
Не зависит от EventBus — это чистые функции/обёртки над существующим кодом.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from ..pipeline.hypotheses import HypothesisSpec, generate_hypotheses
from ..pipeline.planner import ChatPlanner, ResearchPlan
from ..types import BacktestResult, PipelineResult
from ..validation.cpcv import CombinatorialPurgedCV
from ..validation.deflated_sharpe import deflated_sharpe_ratio
from ..validation.pbo import probability_of_backtest_overfitting

logger = logging.getLogger(__name__)

# ── plan_research ───────────────────────────────────────────────

async def plan_research(goal: str) -> dict[str, Any]:
    """Разобрать цель пользователя на исполнимый ResearchPlan.

    Дополнительно проверяет дедупликацию по embedding: если в реестре уже есть
    семантически близкие гипотезы (cosine ≥ 0.92) — добавляет в план поля
    `similar_runs` и `dedup_warning` для агента.
    """
    planner = ChatPlanner()
    plan = planner.plan(goal)
    result = asdict(plan)

    # Дедупликация через embedding
    from ..db import _async_session_factory
    from ..registry.embeddings import Embedder
    from ..registry.store import RegistryStore

    candidates_text = Embedder.hypothesis_to_text(
        family=",".join(plan.hypothesis_families) or "unknown",
        ticker=",".join(plan.tickers) or "unknown",
        params={"start": plan.start_date, "end": plan.end_date},
    )
    embedder = Embedder()
    emb = await embedder.embed(candidates_text)
    async with _async_session_factory() as db:
        store = RegistryStore(db)
        similar = await store.search_similar(emb, threshold=0.92, limit=5)
    if similar:
        result["similar_runs"] = [
            {
                "family": h.family, "ticker": h.ticker,
                "similarity": round(sim, 3),
                "previous_dsr": h.dsr,
                "previous_sharpe": h.sharpe,
                "run_id": str(h.run_id),
            }
            for h, sim in similar
        ]
        result["dedup_warning"] = (
            f"Похожие гипотезы уже проверялись: "
            f"{', '.join(f'{h.family}/{h.ticker}' for h, _ in similar[:3])}. "
            f"Возможно, следует сменить параметры или тикер."
        )

    return result


# ── load_prices ─────────────────────────────────────────────────

async def load_prices(
    tickers: list[str],
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
    timeframe: str = "D1",
) -> dict[str, list[float]]:
    """Загрузить цены закрытия с T-Invest gRPC.

    Read-through кэш через `aqr.data.ohlcv_cache.OhlcvCache`: если в DuckDB-кэше
    уже есть данные для тикера/периода/таймфрейма — сеть не дёргается.
    Для промахов идём в T-Invest gRPC через `TInvestAdapter`, ответ
    складываем в кэш. Без fallback — ошибка propagate'ится.

    Returns: {ticker: [float, ...]} — списки цен закрытия (для JSON-сериализации).
    """
    import asyncio
    import contextlib

    from ..data.ohlcv_cache import OhlcvCache
    from ..data.tinvest import TInvestAdapter

    cache = OhlcvCache()

    # Этап 1: проверить кэш для всех тикеров.
    cached_series: dict[str, pd.Series] = {}
    needs_remote: list[str] = []
    for t in tickers:
        df = await asyncio.to_thread(
            cache.get_cached, t, start_date, end_date, timeframe,
        )
        if df is not None and len(df) >= 100:
            cached_series[t] = df["close"].astype(float)
        else:
            needs_remote.append(t)

    # Этап 2: для промахов идём в T-Invest. Без retry/CB — одна попытка
    # (см. AGENTS.md инвариант 8). Ошибки propagate'ится.
    remote_series: dict[str, pd.Series] = {}
    if needs_remote:
        adapter = TInvestAdapter()

        async def _fetch_one(t: str) -> tuple[str, pd.Series]:
            df = await asyncio.to_thread(
                adapter.candles, t, start_date, end_date, timeframe,
            )
            if len(df) < 100:
                raise ValueError(f"мало данных ({len(df)} строк)")
            with contextlib.suppress(Exception):
                cache.put_cache(t, df, timeframe)
            return t, df["close"].astype(float)

        results = await asyncio.gather(*[_fetch_one(t) for t in needs_remote])
        remote_series = dict(results)

    result = {**cached_series, **remote_series}

    # ── PIT safety net ────────────────────────────────────────────────
    _pit_check_anomalous_returns(result)

    return {t: s.tolist() for t, s in result.items()}


# Threshold для дневного движения, при превышении которого логируем
# warning — типичный сигнал необработанного корпоративного события.
# Для справки: нормальная daily-volatility на MOEX ~1-3%, даже кризисные
# дни редко превышают 10-15%. 20% — высокий порог, чтобы не спамить.
_PIT_RETURN_THRESHOLD = 0.20


def _pit_check_anomalous_returns(prices: dict[str, pd.Series]) -> None:
    """PIT safety net: warning при подозрительных дневных движениях.

    Не вмешивается в данные — только логирует. Полноценный PIT
    (дивиденды, сплиты) — отдельная задача.
    """
    for ticker, series in prices.items():
        if len(series) < 2:
            continue
        rets = series.pct_change().dropna()
        anomalous = rets[rets.abs() > _PIT_RETURN_THRESHOLD]
        if not anomalous.empty:
            worst = anomalous.abs().nlargest(3)
            examples = ", ".join(
                f"{dt.date()}: {r:+.1%}"
                for dt, r in worst.items()
            )
            logger.warning(
                "PIT safety net: %s has %d anomalous daily returns (>%.0f%%) "
                "— possible unprocessed corporate action: %s",
                ticker, len(anomalous), _PIT_RETURN_THRESHOLD * 100, examples,
            )


# ── generate_hypotheses_tool ────────────────────────────────────

async def generate_hypotheses_tool(
    tickers: list[str],
    families: list[str],
    n: int = 20,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Сгенерировать N гипотез по тикерам и семействам."""
    specs = generate_hypotheses(tickers=tickers, families=families, n=n, seed=seed)
    return [
        {
            "name": s.name,
            "family": s.family,
            "ticker": s.ticker,
            "params": s.params,
        }
        for s in specs
    ]


# ── backtest_one ────────────────────────────────────────────────

async def backtest_one(
    hypothesis: dict[str, Any],
    prices: list[float],
    n_hypotheses: int = 20,
    cpcv_splits: int = 6,
    cpcv_test_splits: int = 2,
    embargo_pct: float = 0.01,
) -> dict[str, Any]:
    """Пробэктестировать одну гипотезу: позиция → доходности → Sharpe/DSR/CPCV.

    Args:
        hypothesis: {"family", "ticker", "params", "name"}
        prices: список цен закрытия (должен быть pd.Series-ready)
        n_hypotheses: для поправки DSR на множественное тестирование
    """
    from ..pipeline.hypotheses import make_one_with_params

    # Восстанавливаем HypothesisSpec из параметров (без рандома)
    fam = hypothesis["family"]
    params = hypothesis.get("params", {})
    ticker = hypothesis["ticker"]

    spec = make_one_with_params(fam, ticker, params)
    if spec is None:
        return {"error": f"Неизвестное семейство: {fam}"}

    # Строим pd.Series из списка цен
    price_series = pd.Series(prices, name=ticker)

    # Сигнал → позиции → доходности (позиция сдвигается на 1 бар — без look-ahead)
    pos = spec.fn(price_series)
    pos_shifted = pos.shift(1).fillna(0.0)
    ret = price_series.pct_change().fillna(0.0)
    strat_ret = (pos_shifted * ret).astype(float)
    strat_ret = strat_ret.dropna()

    if len(strat_ret) < 30 or strat_ret.std() == 0:
        return {
            "name": spec.name,
            "family": spec.family,
            "ticker": spec.ticker,
            "params": spec.params,
            "sharpe": 0.0,
            "dsr": 0.0,
            "dsr_verdict": "insufficient",
            "cpcv_mean_sharpe": 0.0,
            "cpcv_std_sharpe": 0.0,
            "max_drawdown": 0.0,
            "n_trades": 0,
            "daily_returns": [],
        }

    sharpe = float(strat_ret.mean() / strat_ret.std() * np.sqrt(252))

    equity = (1 + strat_ret).cumprod()
    dd = float((equity / equity.cummax() - 1.0).min())

    trades = int((pos_shifted.diff().abs() > 0).sum())

    # DSR
    dsr_out = deflated_sharpe_ratio(strat_ret.values, n_trials=n_hypotheses)
    dsr_val = float(dsr_out["deflated_sharpe"])
    if dsr_out["verdict"] == "significant":
        verdict = "significant"
    elif dsr_out["verdict"] == "insufficient_data":
        verdict = "insufficient"
    elif dsr_val >= 0.80:
        verdict = "borderline"
    else:
        verdict = "not_significant"

    # CPCV
    cpcv_mean, cpcv_std = _cpcv_sharpe(strat_ret, cpcv_splits, cpcv_test_splits, embargo_pct)

    return {
        "name": spec.name,
        "family": spec.family,
        "ticker": spec.ticker,
        "params": spec.params,
        "sharpe": round(sharpe, 3),
        "dsr": round(dsr_val, 3),
        "dsr_verdict": verdict,
        "cpcv_mean_sharpe": round(cpcv_mean, 3),
        "cpcv_std_sharpe": round(cpcv_std, 3),
        "max_drawdown": round(dd, 3),
        "n_trades": trades,
        "daily_returns": strat_ret.tolist(),
    }


def _cpcv_sharpe(
    ret: pd.Series,
    n_splits: int = 6,
    n_test_splits: int = 2,
    embargo_pct: float = 0.01,
) -> tuple[float, float]:
    """Средний OOS Sharpe по CPCV путям."""
    try:
        cpcv = CombinatorialPurgedCV(
            n_splits=n_splits,
            n_test_splits=n_test_splits,
            embargo_pct=embargo_pct,
        )
        sharpes = []
        for train_idx, test_idx in cpcv.split(ret.index):
            if len(test_idx) < 20:
                continue
            s = ret.iloc[test_idx]
            if s.std() == 0:
                continue
            sharpes.append(float(s.mean() / s.std() * np.sqrt(252)))
        if not sharpes:
            return 0.0, 0.0
        return float(np.mean(sharpes)), float(np.std(sharpes))
    except Exception:
        return 0.0, 0.0


# ── validate_portfolio ──────────────────────────────────────────

async def validate_portfolio(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """PBO по портфелю результатов бэктеста.

    Args:
        results: список словарей из backtest_one (нужны daily_returns)
    """
    # Извлекаем daily_returns из результатов
    all_returns = []
    for r in results:
        dr = r.get("daily_returns")
        if dr and len(dr) > 0:
            all_returns.append(dr)

    if len(all_returns) < 4:
        return {"pbo": 0.0, "verdict": "insufficient"}

    min_len = min(len(dr) for dr in all_returns)
    min_len = min(min_len, 500)
    if min_len < 40:
        return {"pbo": 0.0, "verdict": "insufficient"}

    M = np.array([dr[-min_len:] for dr in all_returns]).T
    try:
        out = probability_of_backtest_overfitting(M, n_partitions=8)
        return {"pbo": float(out["pbo"]), "verdict": out.get("verdict", "")}
    except Exception:
        return {"pbo": 0.0, "verdict": "error"}


# ── extract_insights ────────────────────────────────────────────

async def extract_insights(
    top_results: list[dict[str, Any]],
    n_tested: int = 0,
    n_survived: int = 0,
    pbo: float = 0.0,
    pbo_verdict: str = "",
) -> list[str]:
    """Детерминистичные наблюдения по топ-результатам."""
    insights: list[str] = []
    if not top_results:
        return insights

    best = top_results[0]
    insights.append(
        f"Лучшая гипотеза: {best.get('family', '?')}: {best.get('name', '?')} "
        f"на {best.get('ticker', '?')}. "
        f"Sharpe={best.get('sharpe', 0):.2f}, DSR={best.get('dsr', 0):.2f} "
        f"({best.get('dsr_verdict', '?')})."
    )

    # По семействам
    by_family: dict[str, list[dict]] = {}
    for r in top_results:
        by_family.setdefault(r.get("family", "?"), []).append(r)
    for fam, xs in by_family.items():
        avg_dsr = np.mean([x.get("dsr", 0) for x in xs])
        insights.append(
            f"Семейство '{fam}' даёт средний DSR {avg_dsr:.2f} на топе "
            f"({len(xs)} гипотез в топ-5)."
        )

    # PBO
    if pbo >= 0.5:
        insights.append(
            f"Внимание: PBO={pbo:.2f} ({pbo_verdict}) — "
            f"портфель гипотез выглядит переобученным."
        )
    elif pbo >= 0.3:
        insights.append(
            f"PBO={pbo:.2f} ({pbo_verdict}) — "
            f"переобучение на грани, перепроверь топ-1 на другом периоде."
        )
    else:
        insights.append(
            f"PBO={pbo:.2f} ({pbo_verdict}) — "
            f"отбор в OOS выглядит устойчивым."
        )

    # Survival rate
    if n_tested > 0:
        rate = n_survived / n_tested
        insights.append(
            f"Выживаемость гипотез после Deflated Sharpe: "
            f"{n_survived}/{n_tested} ({rate:.0%})."
        )

    return insights


# ── review_insights ─────────────────────────────────────────────

async def review_insights(
    goal: str,
    top_results: list[dict[str, Any]],
    deterministic_insights: list[str],
    pbo: float = 0.0,
    pbo_verdict: str = "",
) -> list[str]:
    """LLM-review топ-5 результатов: 0–3 дополнительных наблюдения."""
    from ..pipeline.reviewer import InsightReviewer

    # Минимально восстанавливаем PipelineResult для reviewer
    plan = ResearchPlan(
        goal=goal,
        tickers=list({r.get("ticker", "") for r in top_results}),
        hypothesis_families=list({r.get("family", "") for r in top_results}),
        n_hypotheses=len(top_results),
    )

    top_br = []
    for r in top_results:
        spec = HypothesisSpec(
            name=r.get("name", "?"),
            family=r.get("family", "?"),
            ticker=r.get("ticker", "?"),
            params=r.get("params", {}),
            fn=lambda x: x,  # placeholder — не используется в review
        )
        top_br.append(BacktestResult(
            hypothesis=spec,
            sharpe=r.get("sharpe", 0),
            dsr=r.get("dsr", 0),
            dsr_verdict=r.get("dsr_verdict", "?"),
            cpcv_mean_sharpe=r.get("cpcv_mean_sharpe", 0),
            cpcv_std_sharpe=r.get("cpcv_std_sharpe", 0),
            max_drawdown=r.get("max_drawdown", 0),
            n_trades=r.get("n_trades", 0),
            daily_returns=[],
        ))

    result = PipelineResult(
        run_id="tool-review",
        plan=plan,
        n_hypotheses_tested=len(top_results),
        n_survived_dsr=sum(1 for r in top_results
                           if r.get("dsr_verdict") in ("significant", "borderline")),
        portfolio_pbo=pbo,
        portfolio_pbo_verdict=pbo_verdict,
        top=top_br,
    )

    reviewer = InsightReviewer()
    return reviewer.review(result, deterministic_insights)


# ── narrate ─────────────────────────────────────────────────────

async def narrate(
    goal: str = "",
    tickers: list[str] | None = None,
    families: list[str] | None = None,
    n_tested: int = 0,
    n_survived: int = 0,
    pbo: float = 0.0,
    pbo_verdict: str = "",
    top_results: list[dict[str, Any]] | None = None,
    elapsed_seconds: float = 0.0,
) -> str:
    """Сгенерировать русский отчёт по результатам исследования."""
    from ..pipeline.executor import BacktestResult
    from ..pipeline.narrator import Narrator

    plan = ResearchPlan(
        goal=goal,
        tickers=tickers or [],
        hypothesis_families=families or [],
        n_hypotheses=n_tested,
    )

    top_br = []
    for r in (top_results or []):
        spec = HypothesisSpec(
            name=r.get("name", "?"),
            family=r.get("family", "?"),
            ticker=r.get("ticker", "?"),
            params=r.get("params", {}),
            fn=lambda x: x,
        )
        top_br.append(BacktestResult(
            hypothesis=spec,
            sharpe=r.get("sharpe", 0),
            dsr=r.get("dsr", 0),
            dsr_verdict=r.get("dsr_verdict", "?"),
            cpcv_mean_sharpe=r.get("cpcv_mean_sharpe", 0),
            cpcv_std_sharpe=r.get("cpcv_std_sharpe", 0),
            max_drawdown=r.get("max_drawdown", 0),
            n_trades=r.get("n_trades", 0),
            daily_returns=[],
        ))

    result = PipelineResult(
        run_id="tool-narrate",
        plan=plan,
        n_hypotheses_tested=n_tested,
        n_survived_dsr=n_survived,
        portfolio_pbo=pbo,
        portfolio_pbo_verdict=pbo_verdict,
        top=top_br,
        elapsed_seconds=elapsed_seconds,
    )

    narrator = Narrator()
    return narrator.narrate(result)
