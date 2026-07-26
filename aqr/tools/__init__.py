"""Tool layer: ToolSpec contract + ToolRegistry.

Каждый инструмент = одна асинхронная функция с контрактом ToolSpec.
Чат-агент не знает о пайплайне напрямую — только через реестр инструментов.

WARNING: `registry` ниже — module-level singleton. Тесты должны вызывать
`reset_for_testing()` (или импортировать `reset_registry`) перед `register_all()`,
чтобы избежать дублирования. Под `pytest-xdist` каждый worker имеет свой
singleton (отдельный процесс), но в одном процессе mutation глобальна.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

# Callable signature: async fn(**kwargs) -> Any
ToolFn = Callable[..., Coroutine[Any, Any, Any]]


@dataclass
class ToolSpec:
    """Контракт инструмента: имя, описание для LLM, JSON Schema, функция."""

    name: str
    description: str  # русское описание для LLM-агента
    input_schema: dict[str, Any]  # JSON Schema параметров
    fn: ToolFn
    category: str = "general"  # pipeline | storage | general

    def to_llm_dict(self) -> dict[str, Any]:
        """Сериализация для LLM: только то, что нужно для вызова."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema,
        }


class ToolRegistry:
    """Реестр инструментов. Добавление через register(), поиск через get()."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, tool: ToolSpec) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' уже зарегистрирован")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def list_all(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def list_for_llm(self) -> list[dict[str, Any]]:
        """Список инструментов в формате для LLM-system-prompt."""
        return [t.to_llm_dict() for t in self._tools.values()]

    def reset(self) -> None:
        """Очистить реестр. Используется только в тестах для изоляции."""
        self._tools.clear()

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# Глобальный реестр — синглтон в рамках процесса
registry = ToolRegistry()


def reset_for_testing() -> None:
    """Очистить глобальный реестр. Только для тестов."""
    registry.reset()


__all__ = ["ToolSpec", "ToolRegistry", "ToolFn", "registry", "reset_for_testing"]
