import pandas as pd
import numpy as np

from src.backtest.simple_backtester import sma_crossover_signals, backtest_sma_crossover


def test_signals_basic():
    # Datos sintéticos: subida -> bajada
    prices = [100, 101, 102, 103, 102, 101, 100]
    df = pd.DataFrame({"close": prices})
    out = sma_crossover_signals(df, fast=2, slow=3)
    assert "signal" in out.columns
    # Debe contener al menos alguna señal de entrada o salida
    assert out["signal"].abs().sum() >= 1


def test_backtest_runs():
    np.random.seed(0)
    prices = np.cumsum(np.random.randn(300)) + 100
    df = pd.DataFrame({"close": prices})
    res = backtest_sma_crossover(df, fast=5, slow=20)
    assert "final_return" in res.metrics
    assert isinstance(res.equity_curve, pd.Series)
    # Equity debe tener mismo tamaño que df o casi (por rolling)
    assert len(res.equity_curve) == len(df)
