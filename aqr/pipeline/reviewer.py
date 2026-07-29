"""
InsightReviewer — LLM смотрит на детерминистичные insight'ы и топ-5 результатов
и добавляет 1-3 наблюдения, которые нельзя вытащить шаблоном.

Строгий режим: всегда зовёт LLM с per-session credentials через ContextVar.
Без credentials — raise. На пустом `result.top` — raise.

Живой пример добавленного:
- "Все топ-5 — одно семейство momentum на SBER; edge параметризован узко"
- "n=597 для лучшей гипотезы маловат для DSR — возьми более длинный период"
- "Три из топ-5 — mean_reversion с thr=1.0; на другом пороге стратегия может не работать"
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

    async def review(
        self,
        result: PipelineResult,
        deterministic_insights: list[str],
    ) -> list[str]:
        """Возвращает 0-3 дополнительных инсайта через LLM (async)."""
        if not result.top:
            raise ValueError("InsightReviewer.review: result.top is empty")

        from ..graph.context import current_credentials
        from ..llm_env import acquire_llm_env_lock

        creds = current_credentials()
        if creds is None:
            # REST-путь (без WS-сессии): собираем credentials из env.
            from ..registry.store import DecryptedSettings
            model = self.model or os.environ.get("AQR_LLM_MODEL", "")
            api_key = (
                os.environ.get("DEEPSEEK_API_KEY")
                or os.environ.get("ANTHROPIC_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
                or os.environ.get("GIGACHAT_CREDENTIALS")
                or ""
            )
            if not model or not api_key:
                raise RuntimeError(
                    "InsightReviewer.review: no credentials available."
                )
            creds = DecryptedSettings(
                session_id="rest",
                llm_model=model,
                llm_api_key=api_key,
                openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
                invest_token=os.environ.get("INVEST_TOKEN", ""),
                invest_sandbox=os.environ.get("INVEST_SANDBOX", "1") != "0",
            )

        payload = {
            "goal": result.plan.goal,
            "plan": {
                "tickers": result.plan.tickers,
                "timeframe": result.plan.timeframe,
                "hypothesis_families": result.plan.hypothesis_families,
                "n_hypotheses": result.plan.n_hypotheses,
                "period": f"{result.plan.start_date} → {result.plan.end_date}",
            },
            "n_hypotheses_tested": result.n_hypotheses_tested,
            "n_survived_dsr": result.n_survived_dsr,
            "portfolio_pbo": result.portfolio_pbo,
            "portfolio_pbo_verdict": result.portfolio_pbo_verdict,
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
                for t in result.top
            ],
            "deterministic_insights": deterministic_insights,
        }
        async with await acquire_llm_env_lock() as make_env:
            with make_env(creds):
                import litellm

                resp = await litellm.acompletion(
                    model=creds.llm_model,
                    messages=[
                        {"role": "system", "content": REVIEWER_SYSTEM},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                    response_format={"type": "json_object"},
                )
        if not resp.choices:
            raise RuntimeError("LLM returned no choices for review_insights")
        content = resp.choices[0].message.content
        if content is None:
            raise RuntimeError("LLM returned empty content for review_insights")
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)
        data = json.loads(content)
        obs = data.get("observations", [])
        return [str(o).strip()[:400] for o in obs if o][:3]
