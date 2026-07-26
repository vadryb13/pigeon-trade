"""
InsightReviewer — LLM смотрит на детерминистичные insight'ы и топ-5 результатов
и добавляет 1-3 наблюдения, которые нельзя вытащить шаблоном.

Живой пример добавленного:
- "Все топ-5 — одно семейство momentum на SBER; edge параметризован узко"
- "n=597 для лучшей гипотезы маловат для DSR — возьми более длинный период"
- "Три из топ-5 — mean_reversion с thr=1.0; на другом пороге стратегия может не работать"

Два режима:
- LLM: если задан AQR_LLM_MODEL и есть ключ, зовёт litellm
- Fallback: тихо возвращает [] — детерминистичные insight'ы уже покрыты в executor
"""
from __future__ import annotations

import json
import os

from .executor import PipelineResult

REVIEWER_SYSTEM = """Ты старший quant-ресёрчер. Тебе показали результат прогона:
цель, план, топ-5 гипотез с DSR/Sharpe/n, PBO портфеля, детерминистичные наблюдения.

Твоя задача — добавить 1-3 наблюдения, которых НЕТ в списке detrministic_insights.
Ищи именно то, что шаблон не поймает:
- Concentration risk: топ забит одним тикером/одним семейством/одним параметром
- Данные слабые: маленький n, короткий период
- Подозрительные комбинации: очень высокий Sharpe при маленьком n
- Несоответствие цели и результата: пользователь спросил X, стратегия про Y

Правила:
- Строго по-русски
- Каждое наблюдение — 1 предложение (максимум 2)
- Не повторяй то, что уже есть в detrministic_insights
- Если добавить нечего — верни пустой массив
- Никакого маркетинга, только по существу

Ответь строго валидным JSON: {"observations": ["строка 1", "строка 2", ...]}"""


class InsightReviewer:
    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("AQR_LLM_MODEL")

    def review(
        self,
        result: PipelineResult,
        deterministic_insights: list[str],
    ) -> list[str]:
        """Вернёт 0-3 дополнительных инсайта или пустой список."""
        if not (self.model and self._has_llm_key()):
            return []
        if not result.top:
            return []
        try:
            return self._llm_review(result, deterministic_insights)
        except Exception:
            return []

    def _has_llm_key(self) -> bool:
        keys = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GIGACHAT_CREDENTIALS")
        return any(os.environ.get(k) for k in keys)

    def _llm_review(
        self,
        r: PipelineResult,
        deterministic_insights: list[str],
    ) -> list[str]:
        import litellm

        payload = {
            "goal": r.plan.goal,
            "plan": {
                "tickers": r.plan.tickers,
                "timeframe": r.plan.timeframe,
                "hypothesis_families": r.plan.hypothesis_families,
                "n_hypotheses": r.plan.n_hypotheses,
                "period": f"{r.plan.start_date} → {r.plan.end_date}",
            },
            "n_hypotheses_tested": r.n_hypotheses_tested,
            "n_survived_dsr": r.n_survived_dsr,
            "portfolio_pbo": r.portfolio_pbo,
            "portfolio_pbo_verdict": r.portfolio_pbo_verdict,
            "top_5": [
                {
                    "name": t.hypothesis.describe(),
                    "family": t.hypothesis.family,
                    "ticker": t.hypothesis.ticker,
                    "params": t.hypothesis.params,
                    "sharpe": round(t.sharpe, 2),
                    "dsr": round(t.dsr, 2),
                    "dsr_verdict": t.dsr_verdict,
                    "n_bars": len(t.daily_returns) if t.daily_returns else 0,
                    "n_trades": t.n_trades,
                    "max_drawdown": round(t.max_drawdown, 3),
                    "cpcv_mean_sharpe": round(t.cpcv_mean_sharpe, 2),
                }
                for t in r.top
            ],
            "deterministic_insights": deterministic_insights,
        }
        resp = litellm.completion(
            model=self.model,
            messages=[
                {"role": "system", "content": REVIEWER_SYSTEM},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        obs = data.get("observations", [])
        # Sanity: обрезаем длинные, ограничиваем количество
        return [str(o).strip()[:400] for o in obs if o][:3]
