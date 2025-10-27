import numpy as np
import pandas as pd

from src.ai.features import make_features, split_xy
from src.ai.model import AiModel


def test_features_and_model_train():
    # Datos sintéticos OHLCV
    n = 300
    close = np.cumsum(np.random.randn(n)) + 100
    high = close + np.random.rand(n)
    low = close - np.random.rand(n)
    openp = close + np.random.randn(n) * 0.1
    vol = np.random.rand(n) * 100
    df = pd.DataFrame({
        "open": openp,
        "high": high,
        "low": low,
        "close": close,
        "volume": vol,
    })

    feats = make_features(df)
    X, y = split_xy(feats)
    model = AiModel().fit(X, y)
    proba = model.predict_proba_up(X.tail(10))
    assert proba.shape[0] == 10
    assert np.isfinite(proba).all()
