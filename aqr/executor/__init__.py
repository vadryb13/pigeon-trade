"""NautilusTrader executor for realistic backtesting with fees and slippage.

Usage:
    from aqr.executor.nautilus import execute_with_slippage
    result = await execute_with_slippage(hypothesis={"family": "momentum", ...}, prices=[...])
    # result is a BacktestResult dataclass (from aqr.types)
"""

from __future__ import annotations

from .nautilus import execute_with_slippage

__all__ = ["execute_with_slippage"]
