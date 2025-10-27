from __future__ import annotations

"""Búsqueda de umbrales BUY/SELL para la estrategia IA, con o sin filtro SMA.

Explora una rejilla de valores para maximizar el retorno final con una restricción de drawdown.
Actualiza models/metrics_{symbol}_{tf}.json con el mejor resultado y opcionalmente .env.

Uso:
  python scripts/search_thresholds.py [SYMBOL] [TIMEFRAME] [--no-sma] [--dd-cap 0.15]

Si no se especifica, usa el filtro SMA y parámetros de settings.
"""
import argparse
import json
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd

from src.config.settings import settings
from src.ai.features import make_features, split_xy
from src.ai.model import AiModel
from src.backtest.ai_backtester import backtest_ai


def load_data_and_proba(sym_raw: str, tf: str) -> Tuple[pd.DataFrame, pd.Series]:
    sym = sym_raw.replace("/", "")
    csv_path = Path("data") / f"{sym}_{tf}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No existe CSV {csv_path}. Ejecuta ingesta.")

    df = pd.read_csv(csv_path)
    feats = make_features(df)
    X, _ = split_xy(feats)
    df_aligned = df.iloc[-len(X):].reset_index(drop=True)

    model_path = Path("models") / f"model_{sym}_{tf}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"No existe el modelo {model_path}. Entrena/tunea antes.")

    model = AiModel()
    model.load(model_path)
    proba_up = pd.Series(model.predict_proba_up(X), index=X.index, name="proba_up")
    return df_aligned, proba_up


def search_grid(
    df: pd.DataFrame,
    proba_up: pd.Series,
    use_sma_filter: bool,
    sma_fast: int,
    sma_slow: int,
    dd_cap: float,
) -> dict:
    best = None
    # Rango razonable y con la restricción sell < buy para evitar churn
    buy_grid = np.round(np.arange(0.45, 0.66, 0.03), 2)
    sell_grid = np.round(np.arange(0.30, 0.51, 0.03), 2)

    for b in buy_grid:
        for s in sell_grid:
            if s >= b:
                continue
            res = backtest_ai(
                df,
                proba_up,
                buy_thr=float(b),
                sell_thr=float(s),
                fee=0.0005,
                initial_cash=1000.0,
                sl_pct=settings.risk_stop_loss_pct,
                tp_pct=settings.risk_take_profit_pct,
                trailing_pct=settings.risk_trailing_stop_pct,
                use_sma_filter=use_sma_filter,
                sma_fast=sma_fast,
                sma_slow=sma_slow,
            )
            met = res.metrics
            # max_drawdown es negativo; exigimos que no sea < -dd_cap
            dd_ok = met["max_drawdown"] >= -dd_cap
            if not dd_ok:
                continue
            cand = {
                "buy": float(b),
                "sell": float(s),
                "final_return": float(met["final_return"]),
                "sharpe": float(met["sharpe"]),
                "max_drawdown": float(met["max_drawdown"]),
                "use_sma_filter": bool(use_sma_filter),
                "sma_fast": int(sma_fast),
                "sma_slow": int(sma_slow),
            }
            if best is None or cand["final_return"] > best["final_return"]:
                best = cand
    return best or {}


def persist_best(sym_raw: str, tf: str, best: dict) -> None:
    sym = sym_raw.replace("/", "")
    metrics_path = Path("models") / f"metrics_{sym}_{tf}.json"
    meta = {}
    if metrics_path.exists():
        try:
            meta = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    meta["grid_best"] = best
    metrics_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def update_env(best: dict) -> None:
    # Actualiza .env solo en claves de IA; preserva el resto
    env_path = Path(".env")
    if not env_path.exists():
        return
    lines = env_path.read_text(encoding="utf-8").splitlines()
    keys = {
        "AI_PROBA_BUY": f"AI_PROBA_BUY={best['buy']:.2f}",
        "AI_PROBA_SELL": f"AI_PROBA_SELL={best['sell']:.2f}",
        "AI_SMA_FILTER_USE": f"AI_SMA_FILTER_USE={'true' if best.get('use_sma_filter') else 'false'}",
        "AI_SMA_FAST": f"AI_SMA_FAST={best.get('sma_fast', settings.ai_sma_fast)}",
        "AI_SMA_SLOW": f"AI_SMA_SLOW={best.get('sma_slow', settings.ai_sma_slow)}",
    }
    found = {k: False for k in keys}
    new_lines = []
    for ln in lines:
        wrote = False
        for k, v in keys.items():
            if ln.strip().startswith(f"{k}="):
                new_lines.append(v)
                found[k] = True
                wrote = True
                break
        if not wrote:
            new_lines.append(ln)
    # Agregar claves faltantes al final
    for k, v in keys.items():
        if not found[k]:
            new_lines.append(v)
    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("symbol", nargs="?", default=settings.symbol)
    parser.add_argument("timeframe", nargs="?", default=settings.timeframe)
    parser.add_argument("--no-sma", action="store_true", help="No usar filtro SMA en la búsqueda")
    parser.add_argument("--dd-cap", type=float, default=0.15, help="Límite de drawdown absoluto (p.ej. 0.15)")
    args = parser.parse_args()

    sym_raw = args.symbol
    tf = args.timeframe
    try:
        df, proba_up = load_data_and_proba(sym_raw, tf)
    except Exception as e:
        print(f"[FAIL] {e}")
        return 1

    candidates = []
    # Buscar con SMA si no se desactiva
    if not args["no_sma"] if isinstance(args, dict) else not args.no_sma:
        best_sma = search_grid(
            df,
            proba_up,
            use_sma_filter=True,
            sma_fast=settings.ai_sma_fast,
            sma_slow=settings.ai_sma_slow,
            dd_cap=args.dd_cap,
        )
        if best_sma:
            candidates.append(best_sma)
    # Buscar sin SMA también y comparar
    best_no_sma = search_grid(
        df,
        proba_up,
        use_sma_filter=False,
        sma_fast=settings.ai_sma_fast,
        sma_slow=settings.ai_sma_slow,
        dd_cap=args.dd_cap,
    )
    if best_no_sma:
        candidates.append(best_no_sma)

    if not candidates:
        print("[INFO] No se encontraron combinaciones que cumplan el drawdown cap.")
        return 2

    # Elegir por mayor retorno, desempate por mejor sharpe
    best = sorted(candidates, key=lambda c: (c["final_return"], c["sharpe"]))[-1]
    print("=== Grid Search Best ===")
    print(json.dumps(best, indent=2))

    persist_best(sym_raw, tf, best)
    update_env(best)
    print("[OK] Actualizados metrics y .env con la mejor configuración.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
