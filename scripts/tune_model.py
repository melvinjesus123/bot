"""Búsqueda de hiperparámetros simple para RandomForest usando CV temporal.

Uso:
    python scripts/tune_model.py [SYMBOL] [TIMEFRAME]
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

from src.config.settings import settings
from src.utils.logging_setup import setup_logging
from src.ai.features import make_features, split_xy
from src.ai.model import AiModel, AiModelConfig


def main(symbol: str | None = None, timeframe: str | None = None) -> None:
    logger = setup_logging(settings.log_level)
    sym = (symbol or settings.symbol).replace("/", "")
    tf = timeframe or settings.timeframe
    csv_path = Path("data") / f"{sym}_{tf}.csv"
    if not csv_path.exists():
        logger.error("No existe el CSV: %s. Ejecuta primero la ingesta.", csv_path)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    feats_df = make_features(df)
    X, y = split_xy(feats_df)

    algo = getattr(settings, "ai_algo", "rf").lower()
    if algo == "xgb":
        params_grid = [
            {"algo": "xgb", "xgb_n_estimators": n, "xgb_max_depth": d, "xgb_learning_rate": lr,
             "xgb_subsample": ss, "xgb_colsample_bytree": cs}
            for n in (200, 400, 600)
            for d in (4, 6, 8)
            for lr in (0.05, 0.1, 0.2)
            for ss in (0.8, 0.9, 1.0)
            for cs in (0.8, 0.9, 1.0)
        ]
    else:
        params_grid = [
            {"algo": "rf", "n_estimators": n, "max_depth": d, "class_weight": cw}
            for n in (100, 200, 400)
            for d in (None, 10, 20)
            for cw in (None, "balanced")
        ]

    tscv = TimeSeriesSplit(n_splits=5)
    results = []
    for p in params_grid:
        aucs = []
        for tr_idx, va_idx in tscv.split(X):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
            model = AiModel(AiModelConfig(**p)).fit(X_tr, y_tr)
            proba = model.predict_proba_up(X_va)
            try:
                aucs.append(roc_auc_score(y_va, proba))
            except Exception:
                aucs.append(float('nan'))
        results.append({"params": p, "cv_auc": float(np.nanmean(aucs))})
        logger.info("%s -> cv_auc=%.3f", p, results[-1]["cv_auc"])

    best = max(results, key=lambda r: r["cv_auc"])
    logger.info("Mejores params: %s", best)
    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"tuning_{sym}_{tf}.json", "w", encoding="utf-8") as f:
        json.dump({"results": results, "best": best}, f, ensure_ascii=False, indent=2)

    # Entrena modelo final con mejores params
    best_cfg = AiModelConfig(**best["params"])
    final_model = AiModel(best_cfg).fit(X, y)
    final_model_path = out_dir / f"model_{sym}_{tf}.joblib"
    final_model.save(final_model_path)
    logger.info("Modelo final guardado en %s", final_model_path)


if __name__ == "__main__":
    arg_symbol = sys.argv[1] if len(sys.argv) > 1 else None
    arg_timeframe = sys.argv[2] if len(sys.argv) > 2 else None
    main(arg_symbol, arg_timeframe)
