"""T-Invest MCP Server — JSON-RPC interface for LLM agent data access.

Methods:
  get_candles     — OHLCV from TInvestAdapter
  resolve_figi    — ticker → FIGI
  search_similar  — semantic search past hypotheses (cosine similarity)
  find_duplicates — strict dedup (cosine ≥ 0.92)

Usage:
    from aqr.mcp.server import MCPHandler
    handler = MCPHandler()
    result = await handler.dispatch("get_candles", {"ticker": "SBER"})
"""

from __future__ import annotations

from .protocol import MCPError, MCPRequest, MCPResponse
from .server import MCPHandler, dispatch

__all__ = [
    "MCPHandler",
    "dispatch",
    "MCPRequest",
    "MCPResponse",
    "MCPError",
]
