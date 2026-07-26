"""WebSocket chat слой."""

from .web import router as chat_web_router
from .ws import router as chat_router

__all__ = ["chat_router", "chat_web_router"]
