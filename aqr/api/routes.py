"""FastAPI routes for v0.4 components.

Direct paths (no /v04 prefix):
  POST /team/run          — run_team (5-agent orchestrator)
  POST /executor/nautilus — execute_with_slippage (NautilusTrader)
  POST /mcp/rpc           — MCP JSON-RPC dispatch
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from aqr.auth import require_session_id

logger = logging.getLogger(__name__)

router = APIRouter(tags=["v04"])


class TeamRunRequest(BaseModel):
    goal: str
    tickers: list[str] | None = None
    families: list[str] | None = None


class ExecutorRequest(BaseModel):
    hypothesis: dict[str, Any]
    prices: list[float]
    commission_pct: float = 0.0005
    slippage_ticks: int = 1


class MCPRpcRequest(BaseModel):
    method: str
    params: dict[str, Any] | None = None


@router.post("/team/run")
async def post_team_run(
    req: TeamRunRequest, session_id: str = Depends(require_session_id),
) -> dict[str, Any]:
    """Run the 5-agent team on a research goal."""
    from aqr.agents.orchestrator import run_team

    result = await run_team(
        goal=req.goal,
        session_id=session_id,
        tickers=req.tickers,
        families=req.families,
    )
    return {
        "ok": result.ok,
        "goal": result.goal,
        "summary": result.summary,
        "narrative": result.narrative,
        "insights": result.insights,
        "top_results": result.top_results,
        "n_tested": result.n_tested,
        "n_survived": result.n_survived,
        "pbo": result.validation.get("pbo") if result.validation else None,
        "elapsed_seconds": result.elapsed_seconds,
        "error": result.error,
        "agent_errors": result.agent_errors,
    }


@router.post("/executor/nautilus")
async def post_executor_nautilus(
    req: ExecutorRequest, _session_id: str = Depends(require_session_id),
) -> dict[str, Any]:
    """Run NautilusTrader backtest with realistic execution."""
    from aqr.executor.nautilus import execute_with_slippage

    result = await execute_with_slippage(
        hypothesis=req.hypothesis,
        prices=req.prices,
        commission_pct=req.commission_pct,
        slippage_ticks=req.slippage_ticks,
    )
    return result.to_dict()


@router.post("/mcp/rpc")
async def post_mcp_rpc(
    req: MCPRpcRequest, _session_id: str = Depends(require_session_id),
) -> dict[str, Any]:
    """JSON-RPC dispatch to MCP server."""
    from aqr.mcp.server import dispatch

    return await dispatch(req.method, req.params)
