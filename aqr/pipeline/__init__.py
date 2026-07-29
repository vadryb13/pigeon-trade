"""
Сквозной пайплайн: цель на естественном языке → результат с нарративом.

Модуль спроектирован как минимально работающий вертикальный срез:
- не требует Redis
- не требует DuckDB
- LLM опционален (fallback на детерминистский планировщик)
- все шаги эмитят события в общую очередь для живой ленты
"""
from .events import Event, EventBus
from .executor import PipelineExecutor, PipelineResult
from .narrator import Narrator
from .planner import ResearchPlan, ResearchPlanner
from .reviewer import InsightReviewer

__all__ = [
    "EventBus", "Event",
    "ResearchPlanner", "ResearchPlan",
    "PipelineExecutor", "PipelineResult",
    "Narrator",
    "InsightReviewer",
]
