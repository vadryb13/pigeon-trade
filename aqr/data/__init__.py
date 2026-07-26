"""MOEX ISS adapter with point-in-time guarantees."""

from typing import TYPE_CHECKING, Any

from .moex import MOEXAdapter

__all__ = ["MOEXAdapter", "DataManifest"]


def __getattr__(name: str) -> Any:
    """Ленивый импорт DataManifest — требует duckdb из [data] extra."""
    if name == "DataManifest":
        from .manifest import DataManifest
        return DataManifest
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from .manifest import DataManifest
