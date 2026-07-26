"""AQR v0.4 — new components layered on top of v0.3.

Per the v0.4 plan in AGENTS.md:

- v04.screener   — VectorBT-based fast hypothesis screening
                   (100+ parameter combos in seconds, not minutes)
- v04.executor   — NautilusTrader-based realistic execution
                   (commission, slippage, partial fills)
- v04.agents     — 5-role LangGraph team (Browser / Editor / Analyst /
                   Reviewer / Writer) for parallel research
- v04.mcp        — T-Invest MCP server (replaces direct t_tech.invest
                   calls with JSON-RPC interface for LLM agents)

v0.3 (aqr.pipeline, aqr.agent.graph, aqr.tools, aqr.data) continues to
work as-is. New endpoints wire v0.4 components behind
`aqr/v04/api/routes.py` — v0.3 paths untouched.

Slash-commands: `/run` (v0.3 pipeline), `/team` (v0.4 agents),
`/screener`, `/executor` for explicit v0.4 component access.
"""
__all__: list[str] = []
