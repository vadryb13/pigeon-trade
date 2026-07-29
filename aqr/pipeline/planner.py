"""
ResearchPlanner — принимает свободный запрос пользователя и выдаёт исполнимый ResearchPlan.

Строгий режим: всегда зовёт LLM. Credentials берутся из `current_credentials()`
(per-session ContextVar). Без credentials — RuntimeError.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any


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


PLANNER_SYSTEM = """Ты руководитель quant-исследовательской команды на фондовом рынке.
Пользователь ставит цель на естественном языке. Ты превращаешь её в исполнимый JSON-план.

Правила:
- tickers: список тикеров (российские SBER, GAZP, LKOH и т.д.). Если пользователь сказал "голубые фишки" — SBER, GAZP, LKOH, GMKN, ROSN, TATN. Если "металлурги" — CHMF, NLMK, MAGN, PLZL, GMKN. Если конкретики нет — SBER, GAZP, LKOH.
- timeframe: D1 (день, по умолчанию) | H1 (час) | M60
- start_date / end_date: формат YYYY-MM-DD. По умолчанию последние 2 года.
- hypothesis_families: подмножество [momentum, mean_reversion, breakout, volatility]
- n_hypotheses: 10-50, разумно для запроса
- rationale: 2-3 предложения ПОЧЕМУ такой план

Ответь строго валидным JSON без пояснений."""


class ResearchPlanner:
    """Планировщик: свободный запрос → ResearchPlan через LLM.

    Credentials читаются из per-session ContextVar
    (см. `aqr.agent.context.current_credentials`). Без активной сессии
    с credentials — RuntimeError.
    """

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("AQR_LLM_MODEL")

    async def plan(self, user_goal: str) -> ResearchPlan:
        return await self._llm_plan(user_goal)

    async def _llm_plan(self, user_goal: str) -> ResearchPlan:
        """Зовёт LLM с credentials активной сессии. Raise на любой ошибке."""
        from ..agent.context import current_credentials

        creds = current_credentials()
        if creds is None:
            raise RuntimeError(
                "ResearchPlanner.plan: session credentials not configured. "
                "Open /chat/{token}/settings and enter credentials."
            )

        # Сериализуем env-override через asyncio.Lock — иначе concurrent
        # asyncio.gather-вызовы портят env друг другу в `finally` (B4).
        from ..llm_env import acquire_llm_env_lock

        async with await acquire_llm_env_lock() as make_env:
            with make_env(creds):
                import litellm

                resp = await litellm.acompletion(
                    model=creds.llm_model,
                    messages=[
                        {"role": "system", "content": PLANNER_SYSTEM},
                        {"role": "user", "content": user_goal},
                    ],
                    response_format={"type": "json_object"},
                )
        if not resp.choices:
            raise RuntimeError("LLM returned no choices for plan_research")
        content = resp.choices[0].message.content
        if not content:
            raise RuntimeError("LLM returned empty content for plan_research")
        # Strip markdown code fences if present
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"LLM returned non-JSON response for plan_research: {e}"
            ) from e
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
