"""Health check del bot: valida entorno, datos, modelo y predicción rápida.

Uso:
    python scripts/health_check.py [SYMBOL] [TIMEFRAME]
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src.config.settings import settings
from src.ai.features import make_features, split_xy
from src.ai.model import AiModel


def main(symbol: str | None = None, timeframe: str | None = None) -> int:
    sym_raw = symbol or settings.symbol
    tf = timeframe or settings.timeframe
    sym = sym_raw.replace("/", "")

    print(f"HealthCheck: EXCHANGE={settings.exchange_id} SYMBOL={sym_raw} TF={tf}")

    # 1) CSV de datos
    csv_path = Path("data") / f"{sym}_{tf}.csv"
    if not csv_path.exists():
        print(f"[FAIL] No existe CSV {csv_path}. Ejecuta la ingesta (Ingest forever/once).")
        return 1
    df = pd.read_csv(csv_path)
    if df.empty or "close" not in df.columns:
        print("[FAIL] CSV vacío o sin columna 'close'.")
        return 1
    print(f"[OK] CSV encontrado: {csv_path} | filas={len(df)}")

    # 2) Modelo
    model_path = Path("models") / f"model_{sym}_{tf}.joblib"
    if not model_path.exists():
        print(f"[FAIL] No existe el modelo {model_path}. Entrena o ejecuta tuning.")
        return 1
    model = AiModel()
    model.load(model_path)
    print(f"[OK] Modelo cargado: {model_path}")

    # 3) Features y predicción
    feats = make_features(df)
    X, _ = split_xy(feats)
    if X.empty:
        print("[FAIL] Features vacías para predicción.")
        return 1
    proba_up = float(model.predict_proba_up(X.tail(1))[0])
    print(f"[OK] Predicción rápida: proba_up={proba_up:.3f}")

    # 4) Reglas básicas activas
    print(
        "[OK] Riesgo: SL={:.2%}, TP={:.2%}, TR={:.2%}, MaxDailyLoss={:.2%}, MaxConsecLosses={}, KillDD={:.2%}".format(
            settings.risk_stop_loss_pct,
            settings.risk_take_profit_pct,
            settings.risk_trailing_stop_pct,
            settings.risk_max_daily_loss_pct,
            settings.risk_max_consecutive_losses,
            settings.risk_kill_switch_drawdown_pct,
        )
    )

    print("[SUCCESS] Health check completo.")
    return 0


if __name__ == "__main__":
    arg_symbol = sys.argv[1] if len(sys.argv) > 1 else None
    arg_timeframe = sys.argv[2] if len(sys.argv) > 2 else None
    raise SystemExit(main(arg_symbol, arg_timeframe))
