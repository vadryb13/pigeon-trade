"""MCP Server — dispatches JSON-RPC methods to internal AQR services.

Each method is an async handler that reads per-session credentials
from `aqr.agent.context.current_credentials()`.

Supported methods:
  get_candles(ticker, from_date, to_date, interval) — OHLCV from TInvestAdapter
  resolve_figi(ticker) — ticker → FIGI string
  search_similar(text, threshold, limit) — semantic search
  find_duplicates(text, threshold) — strict dedup
"""

from __future__ import annotations

import logging
from typing import Any

from .protocol import (
    METHOD_NOT_FOUND,
    MCPError,
    error_response,
    success_response,
)

logger = logging.getLogger(__name__)


class MCPHandler:
    """Dispatches JSON-RPC method calls to the appropriate handler.

    Usage:
        handler = MCPHandler()
        result = await handler.dispatch("get_candles", {"ticker": "SBER"})
    """

    def __init__(self) -> None:
        self._methods: dict[str, Any] = {
            "get_candles": self._get_candles,
            "resolve_figi": self._resolve_figi,
            "search_similar": self._search_similar,
            "find_duplicates": self._find_duplicates,
        }

    async def dispatch(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        req_id: int | str | None = None,
    ) -> dict[str, Any]:
        """Dispatch a JSON-RPC method call.

        Returns a JSON-RPC 2.0 response dict (success or error).
        """
        handler = self._methods.get(method)
        if handler is None:
            return error_response(METHOD_NOT_FOUND, req_id)

        try:
            result = await handler(**(params or {}))
            return success_response(result, req_id)
        except TypeError as exc:
            logger.exception("Invalid params for %s", method)
            return error_response(
                MCPError(code=-32602, message=f"Invalid params: {exc}"),
                req_id,
            )
        except Exception:
            logger.exception("MCP method %s failed", method)
            return error_response(
                MCPError(code=-32603, message="Internal error"),
                req_id,
            )

    # ── Method handlers ─────────────────────────────────────────

    async def _get_candles(
        self,
        ticker: str,
        from_date: str = "2023-01-01",
        to_date: str = "2024-12-31",
        interval: str = "D1",
    ) -> dict[str, Any]:
        """Get OHLCV data for a ticker.

        Delegates to TInvestAdapter.candles.
        Returns {close: [...], open: [...], high: [...], low: [...], volume: [...]}.
        """
        from aqr.data.tinvest import TInvestAdapter

        adapter = TInvestAdapter()
        df = await adapter.candles(ticker, from_date, to_date, interval)
        return {
            col: df[col].tolist()
            for col in ["open", "high", "low", "close", "volume"]
        }

    async def _resolve_figi(self, ticker: str) -> str:
        """Resolve ticker → FIGI."""
        from aqr.data.tinvest import TInvestAdapter

        adapter = TInvestAdapter()
        return await adapter._resolve_figi(ticker)

    async def _search_similar(
        self,
        text: str,
        threshold: float = 0.85,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Semantic search for similar hypotheses.

        Uses OpenAI embeddings + pgvector cosine similarity.
        """
        from aqr.session import async_session_factory
        from aqr.registry.embeddings import Embedder
        from aqr.registry.store import RegistryStore

        embedder = Embedder()
        emb = await embedder.embed(text)

        async with async_session_factory() as db:
            store = RegistryStore(db)
            similar = await store.search_similar(emb, threshold=threshold, limit=limit)

        return [
            {
                "family": h.family,
                "ticker": h.ticker,
                "similarity": round(sim, 3),
                "dsr": h.dsr,
                "sharpe": h.sharpe,
                "run_id": str(h.run_id),
            }
            for h, sim in similar
        ]

    async def _find_duplicates(
        self,
        text: str,
        threshold: float = 0.92,
    ) -> list[dict[str, Any]]:
        """Find near-exact duplicates among past hypotheses."""
        return await self._search_similar(text, threshold=threshold, limit=10)


async def dispatch(
    method: str,
    params: dict[str, Any] | None = None,
    req_id: int | str | None = None,
) -> dict[str, Any]:
    """Convenience: create handler and dispatch in one call."""
    handler = MCPHandler()
    return await handler.dispatch(method, params, req_id)
