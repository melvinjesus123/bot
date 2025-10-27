from __future__ import annotations
"""Chequeo de preparación para operar en real con ccxt.

Valida:
- EXCHANGE_ID y credenciales (.env)
- Conexión a la API (time, markets)
- Mercado del símbolo actual: precisión, límites, min notional/cantidad
- Ticker y balances básicos

Uso:
  python scripts/live_readiness_check.py [SYMBOL]
"""
import sys
import json
from pathlib import Path

import ccxt  # type: ignore

from src.config.settings import settings


def main(arg_symbol: str | None = None) -> int:
    sym = (arg_symbol or settings.symbol).upper()
    ex_id = settings.exchange_id

    print(f"Exchange: {ex_id} | Symbol: {sym}")

    # Si estamos en PAPER_TRADING, no exigimos credenciales y devolvemos éxito amigable.
    if settings.paper_trading and (not settings.api_key or not settings.api_secret):
        print("[OK] PAPER_TRADING=true: no se requieren credenciales para este chequeo.")
        print("      Para LIVE, completa API_KEY/API_SECRET en .env y vuelve a ejecutar el chequeo.")
        return 0
    if not settings.api_key or not settings.api_secret:
        print("[WARN] API_KEY/API_SECRET no configurados en .env — no se puede operar en real.")
        return 2

    exchange_class = getattr(ccxt, ex_id)
    opts = {
        "enableRateLimit": True,
        "apiKey": settings.api_key,
        "secret": settings.api_secret,
    }
    if settings.api_passphrase:
        opts["password"] = settings.api_passphrase

    ex = exchange_class(opts)

    try:
        srv_time = ex.milliseconds()
        print(f"[OK] Acceso base a API (milliseconds): {srv_time}")
    except Exception as e:
        print(f"[FAIL] No se pudo acceder a la API base: {e}")
        return 1

    try:
        markets = ex.load_markets()
        if sym not in markets:
            print(f"[FAIL] Símbolo {sym} no encontrado en markets.")
            return 1
        m = markets[sym]
        print("[OK] Mercado cargado | info básica:")
        info = {
            "symbol": m.get("symbol"),
            "precision": m.get("precision"),
            "limits": m.get("limits"),
            "taker": m.get("taker"),
            "maker": m.get("maker"),
        }
        print(json.dumps(info, indent=2))
        # Min notional si está disponible
        min_cost = None
        limits = m.get("limits") or {}
        cost = limits.get("cost") or {}
        min_cost = cost.get("min")
        if min_cost:
            print(f"[INFO] Min notional exchange: {min_cost}")
        else:
            print("[INFO] Min notional no provisto por el exchange en metadata; usar LIVE_MIN_NOTIONAL_USD.")
    except Exception as e:
        print(f"[FAIL] Error cargando markets: {e}")
        return 1

    try:
        ticker = ex.fetch_ticker(sym)
        last = ticker.get("last") or ticker.get("close")
        print(f"[OK] Ticker: last={last}")
    except Exception as e:
        print(f"[FAIL] No se pudo obtener ticker: {e}")
        return 1

    try:
        bal = ex.fetch_balance()
        base, quote = sym.split("/")
        base_free = bal.get(base, {}).get("free", 0.0)
        quote_free = bal.get(quote, {}).get("free", 0.0)
        print(f"[OK] Balances | {base} free={base_free} | {quote} free={quote_free}")
    except Exception as e:
        print(f"[WARN] No se pudo obtener balances: {e}")

    print("[SUCCESS] Chequeo LIVE completo. Cuando quieras, establece PAPER_TRADING=false y reinicia el motor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
