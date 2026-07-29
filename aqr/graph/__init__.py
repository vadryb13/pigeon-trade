"""Agent layer: LangGraph-based chat agent that orchestrates tools.

Public API:
    from aqr.graph import run_agent       # high-level helper
    from aqr.graph import get_agent       # compiled singleton graph
    from aqr.graph import SessionContext  # history + best strategy + untested combos
"""
from .context import SessionContext
from .graph import get_agent, run_agent

__all__ = ["run_agent", "get_agent", "SessionContext"]
