from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class AiBacktestResult:
    metrics: Dict[str, float]
    equity_curve: pd.Series
    trades: pd.DataFrame


def backtest_ai(
    df: pd.DataFrame,
    proba_up: pd.Series,
    buy_thr: float,
    sell_thr: float,
    *,
    fee: float = 0.0005,
    initial_cash: float = 1000.0,
    sl_pct: float = 0.02,
    tp_pct: float = 0.04,
    trailing_pct: float = 0.0,
    use_sma_filter: bool = False,
    sma_fast: int = 20,
    sma_slow: int = 50,
) -> AiBacktestResult:
    """Backtest IA long-only basado en probabilidad de subida.

    Reglas:
    - Entra long cuando proba_up >= buy_thr y no hay posición.
    - Sale cuando proba_up <= sell_thr o por SL/TP/trailing.
    - Comisiones aplicadas a la entrada/salida.
    """
    if "close" not in df.columns:
        raise ValueError("El DataFrame requiere la columna 'close'")

    proba_up = proba_up.reindex(df.index)
    position = 0
    cash = initial_cash
    qty = 0.0
    entry_price: Optional[float] = None
    high_price: Optional[float] = None

    equity_points = []
    trades = []

    # Precalcular SMA si se usa filtro
    if use_sma_filter:
        df = df.copy()
        df["sma_fast"] = df["close"].rolling(sma_fast, min_periods=sma_fast).mean()
        df["sma_slow"] = df["close"].rolling(sma_slow, min_periods=sma_slow).mean()

    for idx, row in df.iterrows():
        price = float(row["close"]) if not np.isnan(row["close"]) else None
        p_up = float(proba_up.loc[idx]) if not np.isnan(proba_up.loc[idx]) else None
        if price is None or p_up is None:
            if equity_points:
                equity_points.append((idx, equity_points[-1][1]))
            else:
                equity_points.append((idx, initial_cash))
            continue

        # Actualizar trailing high
        if position == 1 and high_price is not None:
            high_price = max(high_price, price)

        # Stops primero
        closed_by_risk = False
        if position == 1 and entry_price:
            # Trailing stop
            if trailing_pct > 0 and high_price:
                trail = high_price * (1 - trailing_pct)
                if price <= trail:
                    cash = qty * price * (1 - fee)
                    trades.append({"time": idx, "side": "risk-exit", "price": price, "qty": qty})
                    qty = 0.0
                    position = 0
                    entry_price = None
                    high_price = None
                    closed_by_risk = True
            # SL/TP
            if not closed_by_risk and entry_price:
                if price <= entry_price * (1 - sl_pct) or price >= entry_price * (1 + tp_pct):
                    cash = qty * price * (1 - fee)
                    trades.append({"time": idx, "side": "risk-exit", "price": price, "qty": qty})
                    qty = 0.0
                    position = 0
                    entry_price = None
                    high_price = None
                    closed_by_risk = True

        # Señales IA (si no se cerró por riesgo arriba)
        if not closed_by_risk:
            trend_ok = True
            if use_sma_filter:
                sf = row.get("sma_fast", np.nan)
                ss = row.get("sma_slow", np.nan)
                trend_ok = bool(not np.isnan(sf) and not np.isnan(ss) and sf > ss)

            if position == 0 and p_up >= buy_thr and trend_ok:
                # buy
                qty = (cash * (1 - fee)) / price
                cash = 0.0
                position = 1
                entry_price = price
                high_price = price
                trades.append({"time": idx, "side": "buy", "price": price, "qty": qty})
            elif position == 1 and (p_up <= sell_thr or (use_sma_filter and not trend_ok)):
                # sell
                cash = qty * price * (1 - fee)
                trades.append({"time": idx, "side": "sell", "price": price, "qty": qty})
                qty = 0.0
                position = 0
                entry_price = None
                high_price = None

        eq = cash + qty * price
        equity_points.append((idx, eq))

    equity_series = pd.Series([v for _, v in equity_points], index=[i for i, _ in equity_points], name="equity")
    # métricas
    final_ret = (equity_series.iloc[-1] / initial_cash) - 1 if len(equity_series) else 0.0
    returns = equity_series.pct_change().dropna()
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() and len(returns) > 0 else 0.0
    roll_max = equity_series.cummax()
    drawdown = (equity_series / roll_max - 1.0)
    max_dd = drawdown.min() if len(drawdown) else 0.0

    return AiBacktestResult(
        metrics={"final_return": float(final_ret), "sharpe": float(sharpe), "max_drawdown": float(max_dd)},
        equity_curve=equity_series,
        trades=pd.DataFrame(trades),
    )
