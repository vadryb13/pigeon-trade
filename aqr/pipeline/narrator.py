"""
Narrator — превращает сырой PipelineResult в человекочитаемый рассказ.

Два режима:
- LLM: если задан AQR_LLM_MODEL и есть ключ, пишет через litellm
- Fallback: детерминистский шаблон, всё равно живой русский текст
"""
from __future__ import annotations

import json
import os

from .executor import PipelineResult

NARRATOR_SYSTEM = """Ты quant-исследователь, отчитывающийся коллеге о ночной проверке гипотез.
Говори по-русски, кратко, по делу, без маркетинга. Пиши повествованием, а не таблицами.

Что важно упомянуть:
- Что было целью
- Сколько гипотез проверил и по каким тикерам
- Какая гипотеза лучшая, какой у неё Deflated Sharpe и что это значит
- Как выглядит PBO (переобучение) и что это значит для доверия к результату
- Если ничего значимого не нашёл — сказать честно
- Ограничения: маленький сэмпл, синтетические данные, короткий период — если применимо

3-6 абзацев. Никаких emoji, никаких списков, только связный текст."""


class Narrator:
    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("AQR_LLM_MODEL")

    def narrate(self, result: PipelineResult) -> str:
        if self.model and self._has_llm_key():
            try:
                return self._llm_narrate(result)
            except Exception:
                pass
        return self._fallback_narrate(result)

    def _has_llm_key(self) -> bool:
        keys = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GIGACHAT_CREDENTIALS")
        return any(os.environ.get(k) for k in keys)

    def _llm_narrate(self, result: PipelineResult) -> str:
        import litellm
        payload = {
            "goal": result.plan.goal,
            "tickers": result.plan.tickers,
            "timeframe": result.plan.timeframe,
            "n_tested": result.n_hypotheses_tested,
            "n_survived_dsr": result.n_survived_dsr,
            "portfolio_pbo": result.portfolio_pbo,
            "portfolio_pbo_verdict": result.portfolio_pbo_verdict,
            "top": [r.to_dict() for r in result.top],
            "elapsed_seconds": result.elapsed_seconds,
        }
        resp = litellm.completion(
            model=self.model,
            messages=[
                {"role": "system", "content": NARRATOR_SYSTEM},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        return resp.choices[0].message.content.strip()

    def _fallback_narrate(self, r: PipelineResult) -> str:
        p = r.plan
        parts: list[str] = []

        parts.append(
            f"Разбирал вопрос: «{p.goal}». "
            f"Взял {len(p.tickers)} тикеров ({', '.join(p.tickers)}) "
            f"на таймфрейме {p.timeframe}, семейства гипотез — {', '.join(p.hypothesis_families)}. "
            f"Всего проверил {r.n_hypotheses_tested} гипотез, работа заняла "
            f"{r.elapsed_seconds:.1f} секунд."
        )

        if not r.top:
            parts.append(
                "Ничего проверить не удалось: либо данные не загрузились, "
                "либо гипотезы не сгенерировались. Смотри лог событий."
            )
            return "\n\n".join(parts)

        best = r.top[0]
        parts.append(
            f"Лучшая гипотеза — {best.hypothesis.describe()}. "
            f"Наблюдаемый годовой Sharpe {best.sharpe:.2f}, "
            f"после поправки на множественное тестирование "
            f"(Deflated Sharpe = {best.dsr:.2f}) вердикт «{self._verdict_ru(best.dsr_verdict)}». "
            f"На CPCV OOS-путях средний Sharpe {best.cpcv_mean_sharpe:.2f} "
            f"со стандартным отклонением {best.cpcv_std_sharpe:.2f} — это ориентир, "
            f"как стратегия ведёт себя на невиденных участках."
        )

        if r.n_survived_dsr == 0:
            parts.append(
                "Ни одна гипотеза не прошла порог Deflated Sharpe. "
                "Учитывая, что я тестировал "
                f"{r.n_hypotheses_tested} штук одновременно, планка была высокой: "
                f"любой Sharpe надо смотреть с учётом того, что часть из "
                "них случайно окажется высокой на шуме."
            )
        else:
            parts.append(
                f"Через фильтр Deflated Sharpe прошло {r.n_survived_dsr} гипотез из {r.n_hypotheses_tested} "
                f"({r.n_survived_dsr/max(r.n_hypotheses_tested,1):.0%}). "
                "Это те, которые статистически значимо превышают ожидаемый максимум по шуму "
                "при данном количестве попыток."
            )

        pbo_ru = self._verdict_ru(r.portfolio_pbo_verdict)
        if r.portfolio_pbo >= 0.5:
            parts.append(
                f"Тревожный сигнал по портфелю: PBO = {r.portfolio_pbo:.2f} ({pbo_ru}). "
                "Это значит, что стратегия, победившая на исторических данных, "
                "с большой вероятностью окажется хуже медианы на out-of-sample. "
                "Верхние Sharpe стоит перепроверить на другом периоде или дополнительном таймфрейме."
            )
        elif r.portfolio_pbo >= 0.3:
            parts.append(
                f"По портфелю PBO = {r.portfolio_pbo:.2f} ({pbo_ru}) — "
                "переобучение на грани. Скорее всего лучшая стратегия останется в верхней половине и на OOS, "
                "но топ-1 стоит перепроверить на другом периоде или другом таймфрейме."
            )
        else:
            parts.append(
                f"По портфелю PBO = {r.portfolio_pbo:.2f} ({pbo_ru}) — "
                "отбор лучшей стратегии выглядит устойчивым, "
                "победитель на in-sample скорее всего останется в верхней половине и на out-of-sample."
            )

        # Ограничения
        limits = []
        if r.n_hypotheses_tested < 15:
            limits.append("маленький сэмпл гипотез")
        if p.timeframe == "H1":
            limits.append("часовые данные хорошо покрываются MOEX ISS только на коротком горизонте")
        if limits:
            parts.append("Ограничения: " + ", ".join(limits) + ".")

        return "\n\n".join(parts)

    def _verdict_ru(self, v: str) -> str:
        return {
            "significant": "значимо",
            "borderline": "на грани",
            "not_significant": "незначимо",
            "insufficient": "мало данных",
            "insufficient_data": "мало данных",
            "overfit": "переобучен",
            "suspicious": "подозрительно",
            "robust": "устойчиво",
            "error": "ошибка",
        }.get(v, v)
