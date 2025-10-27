"""Entrena un modelo IA con datos OHLCV locales y guarda en models/.

Uso:
    python scripts/train_model.py [SYMBOL] [TIMEFRAME]
Por defecto usa SYMBOL y TIMEFRAME del .env
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
import json
import numpy as np

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
    if len(feats_df) < 200:
        logger.error("Muy pocos datos para entrenar: %d filas", len(feats_df))
        sys.exit(1)

    # CV temporal para evaluar estabilidad
    X_all, y_all = split_xy(feats_df)
    tscv = TimeSeriesSplit(n_splits=5)
    aucs = []
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_all), 1):
        X_tr, X_va = X_all.iloc[tr_idx], X_all.iloc[va_idx]
        y_tr, y_va = y_all.iloc[tr_idx], y_all.iloc[va_idx]
        model_cv = AiModel(AiModelConfig(algo=getattr(settings, "ai_algo", "rf"))).fit(X_tr, y_tr)
        proba_cv = model_cv.predict_proba_up(X_va)
        try:
            aucs.append(roc_auc_score(y_va, proba_cv))
        except Exception:
            aucs.append(float('nan'))
    logger.info("CV AUC por fold: %s", [round(a, 3) for a in aucs])

    # Entrenamiento final con todo el histórico
    model = AiModel(AiModelConfig(algo=getattr(settings, "ai_algo", "rf"))).fit(X_all, y_all)

    # Umbral recomendado (Youden) sobre último 20% como validación hold-out
    split_idx = int(len(X_all) * 0.8)
    X_valid, y_valid = X_all.iloc[split_idx:], y_all.iloc[split_idx:]
    proba = model.predict_proba_up(X_valid)
    thresholds = np.linspace(0.3, 0.7, 41)
    best_thr, best_score = 0.5, -1
    for thr in thresholds:
        preds = (proba >= thr).astype(int)
        try:
            auc = roc_auc_score(y_valid, proba)
            acc = accuracy_score(y_valid, preds)
            youden = acc + auc  # simple criterio combinado
        except Exception:
            youden = -1
        if youden > best_score:
            best_score, best_thr = youden, float(thr)
    logger.info("Umbral recomendado=%.3f (score=%.3f)", best_thr, best_score)

    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"model_{sym}_{tf}.joblib"
    model.save(out_path)
    meta = {
        "symbol": sym,
        "timeframe": tf,
        "features": list(X_all.columns),
        "cv_auc": float(np.nanmean(aucs)) if len(aucs) else float('nan'),
        "thr_recommended": best_thr,
        "algo": getattr(settings, "ai_algo", "rf"),
    }
    with open(out_dir / f"metrics_{sym}_{tf}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info("Modelo guardado en %s y métricas en metrics_%s_%s.json", out_path, sym, tf)


if __name__ == "__main__":
    arg_symbol = sys.argv[1] if len(sys.argv) > 1 else None
    arg_timeframe = sys.argv[2] if len(sys.argv) > 2 else None
    main(arg_symbol, arg_timeframe)
