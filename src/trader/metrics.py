from __future__ import annotations

from pathlib import Path
from typing import Optional
import csv


METRICS_DIR = Path("data/metrics")
TRADES_CSV = METRICS_DIR / "trades.csv"
EQUITY_CSV = METRICS_DIR / "equity.csv"


def _ensure_dir() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)


def append_trade(
    timestamp_iso: str,
    side: str,
    price: float,
    qty: float,
    equity: float,
    pnl_pct: Optional[float] = None,
) -> None:
    _ensure_dir()
    write_header = not TRADES_CSV.exists()
    with TRADES_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "side", "price", "qty", "equity", "pnl_pct"])
        writer.writerow(
            [
                timestamp_iso,
                side,
                f"{price:.8f}",
                f"{qty:.8f}",
                f"{equity:.8f}",
                f"{(pnl_pct if pnl_pct is not None else '')}",
            ]
        )


def append_equity(timestamp_iso: str, equity: float) -> None:
    _ensure_dir()
    write_header = not EQUITY_CSV.exists()
    with EQUITY_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "equity"])
        writer.writerow([timestamp_iso, f"{equity:.8f}"])
