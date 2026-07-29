"""Регистрация всех инструментов в глобальном реестре.

Импортируется при старте приложения — заполняет registry всеми доступными инструментами.
"""

from __future__ import annotations

from . import ToolSpec, registry
from .core import (
    backtest_one,
    extract_insights,
    generate_hypotheses_tool,
    load_prices,
    narrate,
    plan_research,
    review_insights,
    validate_portfolio,
)
from .storage import (
    compare_runs,
    find_duplicates,
    get_run,
    list_runs,
    search_similar_hypotheses,
)

# Expected tool count after registration — increment when adding tools
_EXPECTED_TOOL_COUNT = 13
_registration_done = False


def _safe_register(spec: ToolSpec) -> None:
    if registry.get(spec.name) is None:
        registry.register(spec)


def register_all() -> None:
    """Зарегистрировать все инструменты в глобальном реестре.

    Идемпотентно: повторные вызовы не падают, если инструменты уже зарегистрированы.
    Это позволяет безопасно вызывать register_all() из нескольких точек входа
    (PipelineExecutor.run, agent/graph, тесты).
    """
    global _registration_done
    if _registration_done:
        return
    if len(registry) >= _EXPECTED_TOOL_COUNT:
        _registration_done = True
        return

    # ── Pipeline tools ──────────────────────────────────────────

    _safe_register(ToolSpec(
        name="plan_research",
        description="Разобрать цель пользователя на русском языке в план исследования: "
                    "тикеры, семейства гипотез, временной диапазон, количество гипотез.",
        input_schema={
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "Цель исследования на русском языке"},
            },
            "required": ["goal"],
        },
        fn=plan_research,
        category="pipeline",
    ))

    _safe_register(ToolSpec(
        name="load_prices",
        description="Загрузить дневные цены закрытия для списка тикеров через T-Invest gRPC. "
                    "Read-through DuckDB-кэш; без fallback — ошибка пробрасывается.",
        input_schema={
            "type": "object",
            "properties": {
                "tickers": {"type": "array", "items": {"type": "string"},
                            "description": "Список тикеров"},
                "start_date": {"type": "string", "description": "Дата начала YYYY-MM-DD"},
                "end_date": {"type": "string", "description": "Дата конца YYYY-MM-DD"},
                "timeframe": {"type": "string", "description": "D1 или H1"},
            },
            "required": ["tickers"],
        },
        fn=load_prices,
        category="pipeline",
    ))

    _safe_register(ToolSpec(
        name="generate_hypotheses",
        description="Сгенерировать N гипотез для заданных тикеров и семейств "
                    "(momentum, mean_reversion, breakout, volatility).",
        input_schema={
            "type": "object",
            "properties": {
                "tickers": {"type": "array", "items": {"type": "string"}},
                "families": {"type": "array", "items": {"type": "string"}},
                "n": {"type": "integer", "description": "Количество гипотез"},
            },
            "required": ["tickers", "families"],
        },
        fn=generate_hypotheses_tool,
        category="pipeline",
    ))

    _safe_register(ToolSpec(
        name="backtest_one",
        description="Пробэктестировать одну гипотезу: рассчитать позиции, Sharpe, "
                    "Deflated Sharpe Ratio (DSR), CPCV OOS Sharpe, максимальную просадку.",
        input_schema={
            "type": "object",
            "properties": {
                "hypothesis": {"type": "object", "description": "Словарь гипотезы: family, ticker, params, name"},
                "prices": {"type": "array", "items": {"type": "number"},
                           "description": "Список цен закрытия"},
                "n_hypotheses": {"type": "integer", "description": "Число гипотез в run для DSR"},
                "cpcv_splits": {"type": "integer", "description": "Число фолдов для CPCV"},
                "cpcv_test_splits": {"type": "integer", "description": "Тестовых фолдов в комбинации"},
                "embargo_pct": {"type": "number", "description": "Доля наблюдений для embargo"},
            },
            "required": ["hypothesis", "prices"],
        },
        fn=backtest_one,
        category="pipeline",
    ))

    _safe_register(ToolSpec(
        name="validate_portfolio",
        description="Оценить Probability of Backtest Overfitting (PBO) для портфеля "
                    "результатов бэктеста. PBO > 0.5 → переобучение.",
        input_schema={
            "type": "object",
            "properties": {
                "results": {"type": "array", "items": {"type": "object"},
                            "description": "Список результатов backtest_one"},
            },
            "required": ["results"],
        },
        fn=validate_portfolio,
        category="pipeline",
    ))

    _safe_register(ToolSpec(
        name="extract_insights",
        description="Извлечь детерминистичные наблюдения из топ-результатов: "
                    "лучшая гипотеза, средний DSR по семействам, PBO-вердикт, выживаемость.",
        input_schema={
            "type": "object",
            "properties": {
                "top_results": {"type": "array", "items": {"type": "object"}},
                "n_tested": {"type": "integer"},
                "n_survived": {"type": "integer"},
                "pbo": {"type": "number"},
                "pbo_verdict": {"type": "string"},
            },
            "required": ["top_results"],
        },
        fn=extract_insights,
        category="pipeline",
    ))

    _safe_register(ToolSpec(
        name="review_insights",
        description="LLM-обзор топ-5 результатов: найти дополнительные наблюдения "
                    "(concentration risk, нестабильность параметров, несоответствие цели).",
        input_schema={
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "Исходная цель исследования"},
                "top_results": {"type": "array", "items": {"type": "object"}},
                "deterministic_insights": {"type": "array", "items": {"type": "string"}},
                "pbo": {"type": "number"},
                "pbo_verdict": {"type": "string"},
            },
            "required": ["goal", "top_results", "deterministic_insights"],
        },
        fn=review_insights,
        category="pipeline",
    ))

    _safe_register(ToolSpec(
        name="narrate",
        description="Сгенерировать отчёт на русском языке по результатам исследования: "
                    "3–6 абзацев с резюме цели, лучшей гипотезы, статистики и ограничений.",
        input_schema={
            "type": "object",
            "properties": {
                "goal": {"type": "string"},
                "tickers": {"type": "array", "items": {"type": "string"}},
                "families": {"type": "array", "items": {"type": "string"}},
                "n_tested": {"type": "integer"},
                "n_survived": {"type": "integer"},
                "pbo": {"type": "number"},
                "pbo_verdict": {"type": "string"},
                "top_results": {"type": "array", "items": {"type": "object"}},
                "elapsed_seconds": {"type": "number"},
            },
            "required": ["goal"],
        },
        fn=narrate,
        category="pipeline",
    ))

    # ── Storage tools ───────────────────────────────────────────

    _safe_register(ToolSpec(
        name="get_run",
        description="Получить детали прогона по ID: цель, статус, метрики, топ-гипотезы.",
        input_schema={
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "UUID прогона"},
            },
            "required": ["run_id"],
        },
        fn=get_run,
        category="storage",
    ))

    _safe_register(ToolSpec(
        name="compare_runs",
        description="Сравнить два прогона по метрикам: количество выживших гипотез, PBO, время.",
        input_schema={
            "type": "object",
            "properties": {
                "run_id_a": {"type": "string", "description": "UUID первого прогона"},
                "run_id_b": {"type": "string", "description": "UUID второго прогона"},
            },
            "required": ["run_id_a", "run_id_b"],
        },
        fn=compare_runs,
        category="storage",
    ))

    _safe_register(ToolSpec(
        name="list_runs",
        description="Список последних прогонов в сессии (по умолчанию — последние 10).",
        input_schema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "ID сессии"},
                "limit": {"type": "integer", "description": "Максимум прогонов"},
            },
            "required": [],
        },
        fn=list_runs,
        category="storage",
    ))

    _safe_register(ToolSpec(
        name="search_similar_hypotheses",
        description="Семантический поиск похожих уже-проверенных гипотез по тексту. "
                    "Возвращает top-N с cosine similarity. Использует OpenAI "
                    "text-embedding-3-small или hash-fallback.",
        input_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Описание искомой гипотезы"},
                "threshold": {"type": "number", "description": "Порог cosine similarity 0..1"},
                "limit": {"type": "integer", "description": "Максимум результатов"},
            },
            "required": ["text"],
        },
        fn=search_similar_hypotheses,
        category="storage",
    ))

    _safe_register(ToolSpec(
        name="find_duplicates",
        description="Найти уже-проверенные гипотезы, очень похожие на данный текст "
                    "(cosine ≥ 0.92). Использовать перед запуском прогона, чтобы "
                    "не дублировать прошлые исследования.",
        input_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Описание новой гипотезы"},
                "threshold": {"type": "number", "description": "Порог similarity"},
                "limit": {"type": "integer", "description": "Максимум результатов"},
            },
            "required": ["text"],
        },
        fn=find_duplicates,
        category="storage",
    ))

    _registration_done = True


def _reset_registration_done() -> None:
    """Сбросить флаг регистрации. Только для тестов."""
    global _registration_done
    _registration_done = False
