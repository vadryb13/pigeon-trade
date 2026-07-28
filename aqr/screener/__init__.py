"""VectorBT screener — fast parameter screening for quant hypotheses.

Main entry point: `screen_momentum()` — grid-search SMA-crossover
parameters on a single ticker, returns top-N by Sharpe.
"""
from .vectorbt import VariantResult, screen_momentum

__all__ = ["screen_momentum", "VariantResult"]
