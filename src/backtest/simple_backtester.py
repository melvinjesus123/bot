from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    metrics: Dict[str, float]
    equity_curve: pd.Series
    trades: pd.DataFrame


def sma_crossover_signals(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.DataFrame:
    """Calcula señales SMA crossover. Requiere columna 'close'.

    Señales:
    - +1: cruce alcista (golden cross)
    - -1: cruce bajista (death cross)
    - 0: sin cambio
    """
    out = df.copy()
    out["sma_fast"] = out["close"].rolling(fast, min_periods=fast).mean()
    out["sma_slow"] = out["close"].rolling(slow, min_periods=slow).mean()
    out["signal_raw"] = np.where(out["sma_fast"] > out["sma_slow"], 1, -1)
    out["signal"] = out["signal_raw"].diff().fillna(0)
    out.loc[out.index[0], "signal"] = 0
    # Normaliza: +1 buy entry cuando cruza de -1 a +1; -1 sell/exit cuando +1 a -1
    out["signal"] = out["signal"].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return out


def backtest_sma_crossover(
    df: pd.DataFrame,
    fast: int = 20,
    slow: int = 50,
    fee: float = 0.0005,
    initial_cash: float = 1000.0,
) -> BacktestResult:
    """Backtest simple long-only con SMA crossover.

    Supuestos:
    - Entra long en +1; sale completamente en -1.
    - No usa apalancamiento, 100% del capital disponible en cada entrada.
    - Aplica fee proporcional en cada trade (entrada y salida).
    """
    if "close" not in df.columns:
        raise ValueError("El DataFrame requiere la columna 'close'")
    sig = sma_crossover_signals(df, fast, slow)

    position = 0
    cash = initial_cash
    qty = 0.0
    equity = []
    trades = []

    for idx, row in sig.iterrows():
        price = float(row["close"]) if not np.isnan(row["close"]) else None
        if price is None:
            equity.append((idx, cash + qty * (equity[-1][1] if equity else 0)))
            continue

        # Señales de entrada/salida
        if row["signal"] == 1 and position == 0:
            # buy
            # aplica fee al comprar
            qty = (cash * (1 - fee)) / price
            cash = 0.0
            position = 1
            trades.append({"time": idx, "side": "buy", "price": price, "qty": qty})
        elif row["signal"] == -1 and position == 1:
            # sell
            cash = qty * price * (1 - fee)
            qty = 0.0
            position = 0
            trades.append({"time": idx, "side": "sell", "price": price, "qty": qty})

        # equity mark-to-market
        eq = cash + qty * price
        equity.append((idx, eq))

    equity_series = pd.Series([v for _, v in equity], index=[i for i, _ in equity], name="equity")

    # métricas simples
    ret = (equity_series.iloc[-1] / initial_cash) - 1 if len(equity_series) else 0.0
    returns = equity_series.pct_change().dropna()
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() and len(returns) > 0 else 0.0
    roll_max = equity_series.cummax()
    drawdown = (equity_series / roll_max - 1.0)
    max_dd = drawdown.min() if len(drawdown) else 0.0

    return BacktestResult(
        metrics={"final_return": float(ret), "sharpe": float(sharpe), "max_drawdown": float(max_dd)},
        equity_curve=equity_series,
        trades=pd.DataFrame(trades),
    )
