from __future__ import annotations
# flake8: noqa: E501

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.config.settings import settings

try:
    import xgboost as xgb  # type: ignore
except Exception:  # pragma: no cover - opcional
    xgb = None


@dataclass
class AiModelConfig:
    # Algoritmo: 'rf' o 'xgb'
    algo: str = "rf"
    # RF params
    n_estimators: int = 200
    max_depth: Optional[int] = None
    random_state: int = 42
    class_weight: Optional[str] = None  # e.g., 'balanced'
    # XGB params (si aplica)
    xgb_n_estimators: int = 300
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    xgb_subsample: float = 0.9
    xgb_colsample_bytree: float = 0.9


class AiModel:
    def __init__(self, cfg: AiModelConfig | None = None):
        self.cfg = cfg or AiModelConfig()
        algo = (self.cfg.algo or getattr(settings, "ai_algo", "rf")).lower()
        if algo == "xgb":
            if xgb is None:
                raise ImportError(
                    (
                        "xgboost no está instalado. Añade 'xgboost' a requirements.txt "
                        "e instala las dependencias."
                    )
                )
            self.clf = xgb.XGBClassifier(
                n_estimators=self.cfg.xgb_n_estimators,
                max_depth=self.cfg.xgb_max_depth,
                learning_rate=self.cfg.xgb_learning_rate,
                subsample=self.cfg.xgb_subsample,
                colsample_bytree=self.cfg.xgb_colsample_bytree,
                eval_metric="logloss",
                n_jobs=-1,
                random_state=self.cfg.random_state,
            )
        else:
            self.clf = RandomForestClassifier(
                n_estimators=self.cfg.n_estimators,
                max_depth=self.cfg.max_depth,
                random_state=self.cfg.random_state,
                n_jobs=-1,
                class_weight=self.cfg.class_weight,
            )

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def predict_proba_up(self, X) -> np.ndarray:
        proba = self.clf.predict_proba(X)
        # clase 1 = subida
        if proba.shape[1] == 2:
            return proba[:, 1]
        # fallback para modelos no probabilísticos
        preds = self.clf.predict(X)
        return preds.astype(float)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.clf, path)

    def load(self, path: str | Path) -> None:
        self.clf = joblib.load(path)
