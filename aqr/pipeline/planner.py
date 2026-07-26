"""
ChatPlanner — принимает свободный запрос пользователя и выдаёт исполнимый ResearchPlan.

Работает в двух режимах:
1. LLM-mode: если задан AQR_LLM_MODEL и есть ключ, использует litellm с JSON-schema
2. Fallback-mode: детерминистский парсер по ключевым словам

Fallback покрывает базовые сценарии, чтобы система работала без ключей.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any

# Известные тикеры MOEX (базовое покрытие, для fallback-парсера)
MOEX_TICKERS = {
    "SBER", "GAZP", "LKOH", "GMKN", "ROSN", "TATN", "MTSS", "MGNT",
    "VTBR", "NVTK", "SNGS", "SBERP", "PLZL", "ALRS", "CHMF", "NLMK",
    "MOEX", "YNDX", "OZON", "TCSG", "AFLT", "AFKS", "PHOR", "MAGN",
    "IRAO", "HYDR", "FEES", "RUAL", "MTLR", "POLY", "BSPB", "SIBN",
}


@dataclass
class ResearchPlan:
    """Исполнимый план, который executor может запустить без LLM."""

    goal: str
    tickers: list[str] = field(default_factory=list)
    timeframe: str = "D1"       # D1 | H1 | M60
    start_date: str = "2023-01-01"
    end_date: str = "2024-12-31"
    hypothesis_families: list[str] = field(default_factory=list)
                                 # momentum | mean_reversion | breakout | volatility
    n_hypotheses: int = 20
    validation: dict = field(default_factory=lambda: {
        "cpcv_splits": 6,
        "cpcv_test_splits": 2,
        "embargo_pct": 0.01,
    })
    rationale: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


PLANNER_SYSTEM = """Ты руководитель quant-исследовательской команды на MOEX (Московская биржа).
Пользователь ставит цель на естественном языке. Ты превращаешь её в исполнимый JSON-план.

Правила:
- tickers: список тикеров MOEX (SBER, GAZP, LKOH и т.д.). Если пользователь сказал "голубые фишки" — SBER, GAZP, LKOH, GMKN, ROSN, TATN. Если "металлурги" — CHMF, NLMK, MAGN, PLZL, GMKN. Если конкретики нет — SBER, GAZP, LKOH.
- timeframe: D1 (день, по умолчанию) | H1 (час) | M60
- start_date / end_date: формат YYYY-MM-DD. По умолчанию последние 2 года.
- hypothesis_families: подмножество [momentum, mean_reversion, breakout, volatility]
- n_hypotheses: 10-50, разумно для запроса
- rationale: 2-3 предложения ПОЧЕМУ такой план

Ответь строго валидным JSON без пояснений."""


class ChatPlanner:
    """Планировщик, превращающий свободный запрос в ResearchPlan."""

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("AQR_LLM_MODEL")

    def plan(self, user_goal: str) -> ResearchPlan:
        if self.model and self._has_llm_key():
            try:
                return self._llm_plan(user_goal)
            except Exception:
                # LLM упал — тихо переходим на fallback
                pass
        return self._fallback_plan(user_goal)

    def _has_llm_key(self) -> bool:
        keys = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GIGACHAT_CREDENTIALS")
        return any(os.environ.get(k) for k in keys)

    def _llm_plan(self, user_goal: str) -> ResearchPlan:
        import litellm  # noqa: F401 — import только когда реально используем
        resp = litellm.completion(
            model=self.model,
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM},
                {"role": "user", "content": user_goal},
            ],
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        return self._plan_from_dict(user_goal, data)

    def _plan_from_dict(self, goal: str, data: dict[str, Any]) -> ResearchPlan:
        return ResearchPlan(
            goal=goal,
            tickers=data.get("tickers") or ["SBER", "GAZP", "LKOH"],
            timeframe=data.get("timeframe", "D1"),
            start_date=data.get("start_date", "2023-01-01"),
            end_date=data.get("end_date", "2024-12-31"),
            hypothesis_families=data.get("hypothesis_families")
                or ["momentum", "mean_reversion"],
            n_hypotheses=int(data.get("n_hypotheses", 20)),
            rationale=data.get("rationale", ""),
        )

    # ---------- FALLBACK ----------

    def _fallback_plan(self, goal: str) -> ResearchPlan:
        """Детерминистский парсер: ключевые слова → план."""
        g = goal.lower()

        tickers = self._extract_tickers(g)
        if not tickers:
            tickers = ["SBER", "GAZP", "LKOH"]

        families = []
        if any(w in g for w in ("моментум", "momentum", "тренд", "импульс")):
            families.append("momentum")
        if any(w in g for w in ("возврат", "mean", "reversion", "откат", "разворот")):
            families.append("mean_reversion")
        if any(w in g for w in ("пробой", "breakout", "уровен")):
            families.append("breakout")
        if any(w in g for w in ("волатил", "vol", "спред")):
            families.append("volatility")
        if not families:
            families = ["momentum", "mean_reversion"]

        if any(w in g for w in ("час", "часов", "h1", "hourly", "интрадей", "внутриднев")):
            timeframe = "H1"
        else:
            timeframe = "D1"

        # оценка глубины исследования
        n = 20
        if any(w in g for w in ("быстро", "коротко", "минимум")):
            n = 10
        if any(w in g for w in ("глубоко", "тщательно", "много")):
            n = 40

        rationale = (
            f"Fallback-план (LLM не подключён). Извлёк тикеры {tickers}, "
            f"семейства гипотез {families}, таймфрейм {timeframe}, "
            f"количество гипотез {n}."
        )

        return ResearchPlan(
            goal=goal,
            tickers=tickers,
            timeframe=timeframe,
            hypothesis_families=families,
            n_hypotheses=n,
            rationale=rationale,
        )

    def _extract_tickers(self, g: str) -> list[str]:
        found = []
        # Прямое упоминание тикера в верхнем регистре
        for m in re.findall(r"\b[A-Z]{4,5}\b", g.upper()):
            if m in MOEX_TICKERS and m not in found:
                found.append(m)
        # Русские названия компаний
        aliases = {
            "сбер": "SBER", "сбербанк": "SBER",
            "газпром": "GAZP", "лукойл": "LKOH", "норникел": "GMKN",
            "роснефт": "ROSN", "татнефт": "TATN", "мтс": "MTSS",
            "магнит": "MGNT", "втб": "VTBR", "новатэк": "NVTK",
            "яндекс": "YNDX", "озон": "OZON", "тинькоф": "TCSG",
            "аэрофлот": "AFLT", "полюс": "PLZL", "алроса": "ALRS",
            "северсталь": "CHMF", "нлмк": "NLMK", "мосбирж": "MOEX",
        }
        for ru, tk in aliases.items():
            if ru in g and tk not in found:
                found.append(tk)

        # Категории
        if "голубые фишки" in g or "blue chip" in g:
            for t in ("SBER", "GAZP", "LKOH", "GMKN", "ROSN", "TATN"):
                if t not in found:
                    found.append(t)
        if "металлург" in g:
            for t in ("CHMF", "NLMK", "MAGN", "PLZL", "GMKN"):
                if t not in found:
                    found.append(t)
        if "банк" in g:
            for t in ("SBER", "VTBR", "TCSG", "BSPB"):
                if t not in found:
                    found.append(t)

        return found[:8]  # не больше 8 тикеров
