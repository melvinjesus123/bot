"""Recalibración de umbrales de probabilidad BUY/SELL con hold-out temporal.

Uso:
    python scripts/recalibrate_thresholds.py [SYMBOL] [TIMEFRAME]
Genera/actualiza models/metrics_{SYMBOL}_{TIMEFRAME}.json con campos:
- thr_recommended (BUY)
- thr_sell_recommended (SELL)
- auc_val (AUC en hold-out)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config.settings import settings
from src.ai.features import make_features, split_xy
from src.ai.model import AiModel


def youden_best_threshold(y_true: pd.Series, scores: np.ndarray) -> float:
    # Umbral que maximiza TPR - FPR (Youden J)
    thresholds = np.linspace(0.1, 0.9, 81)
    best_thr = 0.5
    best_j = -1.0
    y = y_true.values.astype(int)
    for thr in thresholds:
        pred = (scores >= thr).astype(int)
        tp = np.sum((pred == 1) & (y == 1))
        fn = np.sum((pred == 0) & (y == 1))
        fp = np.sum((pred == 1) & (y == 0))
        tn = np.sum((pred == 0) & (y == 0))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_thr = float(thr)
    return best_thr


def main(symbol: str | None = None, timeframe: str | None = None) -> int:
    sym_raw = symbol or settings.symbol
    tf = timeframe or settings.timeframe
    sym = sym_raw.replace("/", "")

    csv_path = Path("data") / f"{sym}_{tf}.csv"
    if not csv_path.exists():
        print(f"[FAIL] No existe CSV {csv_path}. Ejecuta la ingesta primero.")
        return 1

    df = pd.read_csv(csv_path)
    feats = make_features(df)
    X, y = split_xy(feats)
    if len(X) < 50:
        print("[FAIL] Muy pocos datos para recalibración.")
        return 1

    # Hold-out temporal: último 20%
    split_idx = int(len(X) * 0.8)
    X_tr, X_va = X.iloc[:split_idx], X.iloc[split_idx:]
    y_tr, y_va = y.iloc[:split_idx], y.iloc[split_idx:]

    model_path = Path("models") / f"model_{sym}_{tf}.joblib"
    if not model_path.exists():
        print(f"[FAIL] No existe el modelo {model_path}. Entrena/tunea antes.")
        return 1

    model = AiModel()
    model.load(model_path)

    proba_va = np.asarray(model.predict_proba_up(X_va))
    # AUC de validación
    try:
        auc_val = float(roc_auc_score(y_va, proba_va))
    except Exception:
        auc_val = float("nan")

    # Umbral BUY sobre proba_up
    thr_buy = youden_best_threshold(y_va, proba_va)
    # Umbral SELL sobre proba_down = 1 - proba_up
    proba_down_va = 1.0 - proba_va
    y_neg = 1 - y_va
    thr_down = youden_best_threshold(y_neg, proba_down_va)
    thr_sell = max(0.0, min(1.0, 1.0 - thr_down))  # traducido a umbral sobre proba_up

    print(f"[INFO] AUC(hold-out)={auc_val:.3f} | thr_buy={thr_buy:.3f} thr_sell={thr_sell:.3f}")

    # Actualizar metrics json
    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"metrics_{sym}_{tf}.json"
    data = {}
    if metrics_path.exists():
        try:
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data["auc_val"] = auc_val
    data["thr_recommended"] = thr_buy
    data["thr_sell_recommended"] = thr_sell
    metrics_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    arg_symbol = sys.argv[1] if len(sys.argv) > 1 else None
    arg_timeframe = sys.argv[2] if len(sys.argv) > 2 else None
    raise SystemExit(main(arg_symbol, arg_timeframe))
