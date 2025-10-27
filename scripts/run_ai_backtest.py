from __future__ import annotations

"""Ejecuta un backtest IA sobre el CSV de datos actual usando el modelo entrenado.

Uso:
    python scripts/run_ai_backtest.py [SYMBOL] [TIMEFRAME] [BUY_THR] [SELL_THR]
Si no se pasan umbrales, usa los de `.env` o, de existir, los recomendados en models/metrics_*.json
"""
import sys
from pathlib import Path

import pandas as pd

from src.config.settings import settings
from src.ai.features import make_features, split_xy
from src.ai.model import AiModel
from src.backtest.ai_backtester import backtest_ai


def resolve_thresholds(sym: str, tf: str, buy_thr: float | None, sell_thr: float | None) -> tuple[float, float]:
    if buy_thr is not None and sell_thr is not None:
        return buy_thr, sell_thr
    metrics_path = Path("models") / f"metrics_{sym}_{tf}.json"
    if metrics_path.exists():
        try:
            import json

            m = json.loads(metrics_path.read_text(encoding="utf-8"))
            if "grid_best" in m:
                gb = m["grid_best"]
                buy_thr = float(gb.get("buy", settings.ai_proba_buy))
                sell_thr = float(gb.get("sell", settings.ai_proba_sell))
            else:
                buy_thr = float(m.get("thr_recommended", settings.ai_proba_buy))
                sell_thr = float(m.get("thr_sell_recommended", settings.ai_proba_sell))
            return buy_thr, sell_thr
        except Exception:
            pass
    return settings.ai_proba_buy, settings.ai_proba_sell


def main(arg_symbol: str | None = None, arg_timeframe: str | None = None, arg_buy: str | None = None, arg_sell: str | None = None) -> int:
    sym_raw = arg_symbol or settings.symbol
    tf = arg_timeframe or settings.timeframe
    sym = sym_raw.replace("/", "")
    buy_thr, sell_thr = resolve_thresholds(sym, tf, float(arg_buy) if arg_buy else None, float(arg_sell) if arg_sell else None)

    csv_path = Path("data") / f"{sym}_{tf}.csv"
    if not csv_path.exists():
        print(f"[FAIL] No existe CSV {csv_path}. Ejecuta ingesta.")
        return 1

    df = pd.read_csv(csv_path)
    feats = make_features(df)
    X, _ = split_xy(feats)
    # Alinear precios con las filas válidas de features (generalmente recorta las primeras filas)
    df_aligned = df.iloc[-len(X):].reset_index(drop=True)

    model_path = Path("models") / f"model_{sym}_{tf}.joblib"
    if not model_path.exists():
        print(f"[FAIL] No existe el modelo {model_path}. Entrena/tunea antes.")
        return 1

    model = AiModel()
    model.load(model_path)
    proba_up = pd.Series(model.predict_proba_up(X), index=X.index, name="proba_up")

    res = backtest_ai(
        df_aligned,
        proba_up,
        buy_thr=buy_thr,
        sell_thr=sell_thr,
        fee=0.0005,
        initial_cash=1000.0,
        sl_pct=settings.risk_stop_loss_pct,
        tp_pct=settings.risk_take_profit_pct,
        trailing_pct=settings.risk_trailing_stop_pct,
        use_sma_filter=settings.ai_sma_filter_use,
        sma_fast=settings.ai_sma_fast,
        sma_slow=settings.ai_sma_slow,
    )

    print("=== AI Backtest Results ===")
    print(f"Symbol: {sym_raw} TF: {tf}")
    print(f"BUY_THR={buy_thr:.2f} SELL_THR={sell_thr:.2f} | SMA_FILTER={settings.ai_sma_filter_use} ({settings.ai_sma_fast}/{settings.ai_sma_slow})")
    print(f"final_return: {res.metrics['final_return']:.2%}")
    print(f"sharpe: {res.metrics['sharpe']:.2f}")
    print(f"max_drawdown: {res.metrics['max_drawdown']:.2%}")
    print(f"trades: {len(res.trades)}")

    # Guardar resumen opcional
    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"ai_backtest_summary_{sym}_{tf}.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"BUY_THR={buy_thr:.2f}",
                f"SELL_THR={sell_thr:.2f}",
                f"final_return={res.metrics['final_return']:.4f}",
                f"sharpe={res.metrics['sharpe']:.4f}",
                f"max_drawdown={res.metrics['max_drawdown']:.4f}",
                f"trades={len(res.trades)}",
            ]
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(
        main(
            sys.argv[1] if len(sys.argv) > 1 else None,
            sys.argv[2] if len(sys.argv) > 2 else None,
            sys.argv[3] if len(sys.argv) > 3 else None,
            sys.argv[4] if len(sys.argv) > 4 else None,
        )
    )
