from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain, index=series.index).rolling(period, min_periods=period).mean()
    roll_down = pd.Series(loss, index=series.index).rolling(period, min_periods=period).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    # Evita FutureWarning usando bfill() en lugar de fillna(method="bfill")
    return out.bfill().fillna(50.0)


def make_features(df: pd.DataFrame, fast: int = 12, slow: int = 26, rsi_period: int = 14) -> pd.DataFrame:
    """Genera un DataFrame de features a partir de OHLCV con columnas esperadas:
    ['open','high','low','close','volume'] y 'timestamp' o 'datetime'.
    """
    x = df.copy()
    # retornos y medias
    x["ret_1"] = x["close"].pct_change()
    x["ret_5"] = x["close"].pct_change(5)
    x["sma_fast"] = x["close"].rolling(fast, min_periods=fast).mean()
    x["sma_slow"] = x["close"].rolling(slow, min_periods=slow).mean()
    x["sma_ratio"] = x["sma_fast"] / x["sma_slow"]
    # EMA y MACD
    x["ema_fast"] = x["close"].ewm(span=fast, adjust=False, min_periods=fast).mean()
    x["ema_slow"] = x["close"].ewm(span=slow, adjust=False, min_periods=slow).mean()
    x["macd"] = x["ema_fast"] - x["ema_slow"]
    x["macd_signal"] = x["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
    x["macd_hist"] = x["macd"] - x["macd_signal"]
    # Bandas de Bollinger
    rolling20 = x["close"].rolling(20, min_periods=20)
    bb_mid = rolling20.mean()
    bb_std = rolling20.std()
    x["bb_upper"] = bb_mid + 2 * bb_std
    x["bb_lower"] = bb_mid - 2 * bb_std
    x["bb_width"] = (x["bb_upper"] - x["bb_lower"]) / bb_mid.replace(0, np.nan)
    # ATR (True Range)
    high_low = (x["high"] - x["low"]).abs()
    high_close = (x["high"] - x["close"].shift()).abs()
    low_close = (x["low"] - x["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    x["atr14"] = tr.rolling(14, min_periods=14).mean()
    x["rsi"] = rsi(x["close"], rsi_period)
    x["vol_roll"] = x["volume"].rolling(10, min_periods=10).mean()
    x["hl_spread"] = (x["high"] - x["low"]) / x["close"].replace(0, np.nan)
    # forward return como objetivo (clasificación binaria subida)
    x["fwd_ret_1"] = x["close"].pct_change(-1) * -1  # equivalente a shift(-1)
    x["target_up"] = (x["fwd_ret_1"] > 0).astype(int)

    feats = [
        "ret_1",
        "ret_5",
        "sma_ratio",
        "ema_fast",
        "ema_slow",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_width",
        "rsi",
        "vol_roll",
        "hl_spread",
        "atr14",
    ]
    x = x.dropna().reset_index(drop=True)
    return x[feats + ["target_up"]]


def split_xy(feats_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = feats_df.drop(columns=["target_up"]).astype(float)
    y = feats_df["target_up"].astype(int)
    return X, y
