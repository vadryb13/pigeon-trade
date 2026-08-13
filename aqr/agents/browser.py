"""BrowserAgent: gathers research context for the team.

Searches:
  - Similar past runs in the registry (dedup)
  - Metadata about requested tickers (name, sector)
  - Any existing hypotheses for the same tickers

Does NOT require LLM — uses DB queries + TInvestAdapter metadata.
"""

from __future__ import annotations

from typing import Any

from .base import AgentResult, BaseAgent


class BrowserAgent(BaseAgent):
    """Researches context: similar runs, ticker metadata, past results."""

    name = "browser"

    async def research(self, goal: str, plan: dict[str, Any] | None = None) -> AgentResult:
        """Gather research context.

        Returns:
            {
                "similar_runs": [...],  # from registry
                "ticker_info": {...},    # resolved ticker → FIGI/name
                "num_similar": int,
                "num_families_tested": int,
            }
        """
        context: dict[str, Any] = {
            "similar_runs": [],
            "ticker_info": {},
            "num_similar": 0,
            "num_families_tested": 0,
        }

        tickers = (plan or {}).get("tickers", [])
        families = (plan or {}).get("hypothesis_families", [])

        # 1) Search similar runs by embedding (if registory available)
        similar = await self._search_similar(goal)
        if similar:
            context["similar_runs"] = similar
            context["num_similar"] = len(similar)

        # 2) Resolve ticker metadata via TInvestAdapter
        info = await self._resolve_ticker_info(tickers)
        context["ticker_info"] = info

        # 3) Count past tests for these families
        tested = await self._count_tested_families(tickers, families)
        context["num_families_tested"] = tested

        return self._ok(**context)

    async def _search_similar(self, text: str) -> list[dict]:
        """Search for similar hypotheses in registry by embedding."""
        from aqr.registry.embeddings import Embedder
        from aqr.registry.store import RegistryStore
        from aqr.session import async_session_factory

        try:
            embedder = Embedder()
            emb = await embedder.embed(text)
        except Exception:
            self.logger.exception("_search_similar: embed failed for %r", text)
            return []

        try:
            async with async_session_factory() as db:
                store = RegistryStore(db)
                similar = await store.search_similar(emb, threshold=0.92, limit=5)
                return [
                    {
                        "family": h.family,
                        "ticker": h.ticker,
                        "similarity": round(sim, 3),
                        "previous_dsr": h.dsr,
                        "run_id": str(h.run_id),
                    }
                    for h, sim in similar
                ]
        except Exception:
            self.logger.exception("_search_similar: DB query failed")
            return []

    async def _resolve_ticker_info(self, tickers: list[str]) -> dict[str, dict]:
        """Resolve ticker → name / sector via TInvestAdapter lazy FIGI."""
        if not tickers:
            return {}
        try:
            from aqr.data.tinvest import TInvestAdapter

            adapter = TInvestAdapter()
            info: dict[str, dict] = {}
            for t in tickers:
                try:
                    figi = await adapter._resolve_figi(t)
                    info[t] = {"figi": figi, "ticker": t}
                except ValueError:
                    info[t] = {"figi": "", "ticker": t, "error": "not found"}
            return info
        except Exception:
            self.logger.exception("_resolve_ticker_info failed")
            return {t: {"ticker": t} for t in tickers}

    async def _count_tested_families(self, tickers: list[str], families: list[str]) -> int:
        """Count how many of the requested (ticker, family) pairs have been tested."""
        if not tickers or not families:
            return 0
        try:
            from aqr.graph.context import SessionContext

            ctx = SessionContext(self.session_id)
            recent = await ctx.get_recent_runs(50)
            if not recent:
                return 0
            # Count unique (family, ticker) pairs across recent runs
            pairs: set[tuple[str, str]] = set()
            run_ids = [r.get("id") for r in recent if r.get("id")]
            from aqr.registry.store import RegistryStore
            from aqr.session import async_session_factory
            async with async_session_factory() as db:
                store = RegistryStore(db)
                by_run = await store.list_hypotheses_by_runs(run_ids)
                for hyps in by_run.values():
                    for h in hyps:
                        pairs.add((h.family, h.ticker))
            return len(pairs)
        except Exception:
            self.logger.exception("_count_tested_families failed")
            return 0
