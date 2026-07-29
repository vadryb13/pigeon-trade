"""Base class for all v0.4 team agents.

Provides credentials access, logging, and error-handling helpers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from aqr.graph.context import current_credentials

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Generic result from any agent node.

    `ok` = True for success, False for error.
    `data` stores the agent's output dict.
    `error` stores an error message if any.
    """

    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""


class BaseAgent:
    """Lightweight base for all team agents.

    Subclasses override `run_agent(input_data) -> AgentResult`.
    """

    name: str = "base"

    def __init__(self, session_id: str = "default") -> None:
        self.session_id = session_id
        self.logger = logging.getLogger(f"aqr.agents.{self.name}")

    @property
    def credentials(self):
        """Shortcut: raises RuntimeError if not set."""
        creds = current_credentials()
        if creds is None:
            raise RuntimeError(
                "Per-session credentials not configured. "
                "Set them via GET /chat/{token}/settings or "
                "aqr.agent.context.set_credentials()."
            )
        return creds

    def _ok(self, **data: Any) -> AgentResult:
        return AgentResult(ok=True, data=data)

    def _fail(self, error: str) -> AgentResult:
        self.logger.error(error)
        return AgentResult(ok=False, error=error)
