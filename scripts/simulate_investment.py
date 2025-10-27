from __future__ import annotations

"""Simula una inversión con capital inicial configurable usando el modelo IA y datos actuales.

- Usa thresholds de models/metrics_{sym}_{tf}.json (grid_best si existe) o .env si no.
- Respeta el filtro SMA según settings.
- Guarda resultados en data/simulations/{sym}_{tf}/{YYYYmmdd_HHMMSS}/trades.csv y equity.csv.
- Imprime resumen con el monto final en USD.

Uso:
  python scripts/simulate_investment.py [CAPITAL_USD] [SYMBOL] [TIMEFRAME]
  Ejemplos:
  python scripts/simulate_investment.py 10
  python scripts/simulate_investment.py 25 ETH/USDT 1h
"""
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple

import pandas as pd

from src.config.settings import settings
from src.ai.features import make_features, split_xy
from src.ai.model import AiModel
from src.backtest.ai_backtester import backtest_ai


def resolve_thresholds(sym: str, tf: str) -> Tuple[float, float]:
    metrics_path = Path("models") / f"metrics_{sym}_{tf}.json"
    if metrics_path.exists():
        try:
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
            if "grid_best" in m:
                gb = m["grid_best"]
                return float(gb.get("buy", settings.ai_proba_buy)), float(gb.get("sell", settings.ai_proba_sell))
            return float(m.get("thr_recommended", settings.ai_proba_buy)), float(m.get("thr_sell_recommended", settings.ai_proba_sell))
        except Exception:
            pass
    return settings.ai_proba_buy, settings.ai_proba_sell


ess = settings


def main() -> int:
    # Parse args
    arg_cash = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    sym_raw = sys.argv[2] if len(sys.argv) > 2 else ess.symbol
    tf = sys.argv[3] if len(sys.argv) > 3 else ess.timeframe
    sym = sym_raw.replace("/", "")

    csv_path = Path("data") / f"{sym}_{tf}.csv"
    if not csv_path.exists():
        print(f"[FAIL] No existe CSV {csv_path}. Ejecuta la ingesta.")
        return 1

    df = pd.read_csv(csv_path)
    feats = make_features(df)
    X, _ = split_xy(feats)
    if X.empty:
        print("[FAIL] Features vacías para backtest.")
        return 1
    df_aligned = df.iloc[-len(X):].reset_index(drop=True)

    model_path = Path("models") / f"model_{sym}_{tf}.joblib"
    if not model_path.exists():
        print(f"[FAIL] No existe el modelo {model_path}. Entrena/tunea antes.")
        return 1

    model = AiModel()
    model.load(model_path)
    proba_up = pd.Series(model.predict_proba_up(X), index=X.index, name="proba_up")

    buy_thr, sell_thr = resolve_thresholds(sym, tf)

    res = backtest_ai(
        df_aligned,
        proba_up,
        buy_thr=buy_thr,
        sell_thr=sell_thr,
        fee=0.0005,
        initial_cash=arg_cash,
        sl_pct=ess.risk_stop_loss_pct,
        tp_pct=ess.risk_take_profit_pct,
        trailing_pct=ess.risk_trailing_stop_pct,
        use_sma_filter=ess.ai_sma_filter_use,
        sma_fast=ess.ai_sma_fast,
        sma_slow=ess.ai_sma_slow,
    )

    # Persistir resultados en carpeta de simulación
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    sim_dir = Path("data") / "simulations" / f"{sym}_{tf}" / ts
    sim_dir.mkdir(parents=True, exist_ok=True)
    trades_path = sim_dir / "trades.csv"
    equity_path = sim_dir / "equity.csv"

    res.trades.to_csv(trades_path, index=False)
    res.equity_curve.rename("equity").to_frame().reset_index(names=["timestamp"]).to_csv(equity_path, index=False)

    final_equity = float(res.equity_curve.iloc[-1]) if len(res.equity_curve) else arg_cash
    summary = {
        "symbol": sym_raw,
        "timeframe": tf,
        "initial_cash": arg_cash,
        "final_equity": final_equity,
        "buy_thr": buy_thr,
        "sell_thr": sell_thr,
        "sma_filter": ess.ai_sma_filter_use,
        "sma_fast": ess.ai_sma_fast,
        "sma_slow": ess.ai_sma_slow,
        "metrics": res.metrics,
    }
    (sim_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== Simulación de inversión ===")
    print(f"Capital inicial: ${arg_cash:.2f}")
    print(f"Símbolo/TF: {sym_raw}/{tf}")
    print(f"Umbrales: BUY>={buy_thr:.2f} SELL<={sell_thr:.2f} | SMA={ess.ai_sma_filter_use} ({ess.ai_sma_fast}/{ess.ai_sma_slow})")
    print(f"Monto final: ${final_equity:.2f}")
    print(f"Carpeta de resultados: {sim_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
