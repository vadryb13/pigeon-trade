"""
Narrator — превращает сырой PipelineResult в человекочитаемый рассказ.

Строгий режим: всегда зовёт LLM с per-session credentials через
ContextVar. Без credentials или при ошибке LLM — raise.
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

    async def narrate(self, result: PipelineResult) -> str:
        return await self._llm_narrate(result)

    async def _llm_narrate(self, result: PipelineResult) -> str:
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
                    "Narrator.narrate: no credentials available. "
                    "Either configure via /chat/{token}/settings or set "
                    "AQR_LLM_MODEL + API key env vars."
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
            "tickers": result.plan.tickers,
            "timeframe": result.plan.timeframe,
            "n_tested": result.n_hypotheses_tested,
            "n_survived_dsr": result.n_survived_dsr,
            "portfolio_pbo": result.portfolio_pbo,
            "portfolio_pbo_verdict": result.portfolio_pbo_verdict,
            "top": [r.to_dict() for r in result.top],
            "elapsed_seconds": result.elapsed_seconds,
        }
        async with await acquire_llm_env_lock() as make_env:
            with make_env(creds):
                import litellm

                resp = await litellm.acompletion(
                    model=creds.llm_model,
                    messages=[
                        {"role": "system", "content": NARRATOR_SYSTEM},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                )
        if not resp.choices:
            raise RuntimeError("LLM returned no choices for narrate")
        content = resp.choices[0].message.content
        if content is None:
            raise RuntimeError("LLM returned empty content for narrate")
        return content.strip()
