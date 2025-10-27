"""Ejecuta un backtest rápido sobre un CSV en data/.

Uso:
    python scripts/run_backtest.py [SYMBOL] [TIMEFRAME]
Por defecto usa SYMBOL y TIMEFRAME del .env
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src.config.settings import settings
from src.backtest.simple_backtester import backtest_sma_crossover
from src.utils.logging_setup import setup_logging


def main(symbol: str | None = None, timeframe: str | None = None) -> None:
    logger = setup_logging(settings.log_level)
    sym = (symbol or settings.symbol).replace("/", "")
    tf = timeframe or settings.timeframe
    csv_path = Path("data") / f"{sym}_{tf}.csv"
    if not csv_path.exists():
        logger.error("No existe el CSV: %s. Ejecuta primero la ingesta.", csv_path)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    res = backtest_sma_crossover(df, fast=20, slow=50)
    logger.info("Backtest: return=%.2f%% | sharpe=%.2f | maxDD=%.2f%%",
                res.metrics["final_return"] * 100,
                res.metrics["sharpe"],
                res.metrics["max_drawdown"] * 100)


if __name__ == "__main__":
    arg_symbol = sys.argv[1] if len(sys.argv) > 1 else None
    arg_timeframe = sys.argv[2] if len(sys.argv) > 2 else None
    main(arg_symbol, arg_timeframe)
