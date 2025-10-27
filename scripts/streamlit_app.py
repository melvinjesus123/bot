from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.config.settings import settings

DATA_DIR = Path("data")
METRICS_DIR = DATA_DIR / "metrics"
SIMULATIONS_DIR = DATA_DIR / "simulations"
TRADES_CSV = METRICS_DIR / "trades.csv"
EQUITY_CSV = METRICS_DIR / "equity.csv"
STATE_JSON = DATA_DIR / "state.json"

# Opcional: fuentes remotas (para Streamlit Cloud u otros despliegues)
# Configúralas en .streamlit/secrets.toml
# Manejo seguro cuando no existe secrets.toml: no romper, usar None.
try:
    METRICS_BASE_URL: Optional[str] = st.secrets.get("METRICS_BASE_URL")  # type: ignore[attr-defined]
except Exception:
    METRICS_BASE_URL = None
try:
    PRICE_CSV_URL: Optional[str] = st.secrets.get("PRICE_CSV_URL")  # type: ignore[attr-defined]
except Exception:
    PRICE_CSV_URL = None


def _read_csv_local_or_url(local_path: Path, url: Optional[str]) -> pd.DataFrame:
    """Lee un CSV desde URL si está configurada; si falla, intenta local.

    Devuelve DataFrame vacío si no existe local y no hay URL válida.
    """
    # Intento remoto
    if url:
        try:
            df = pd.read_csv(url)
            return df
        except Exception:
            pass
    # Fallback local
    if local_path.exists():
        try:
            return pd.read_csv(local_path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def load_price_df() -> pd.DataFrame:
    sym = settings.symbol.replace("/", "")
    csv_path = DATA_DIR / f"{sym}_{settings.timeframe}.csv"
    # Si PRICE_CSV_URL está definida, úsala directamente (o podría ser None)
    df = _read_csv_local_or_url(csv_path, PRICE_CSV_URL)
    if not df.empty and "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df.set_index("datetime", inplace=True)
    return df


ess = st.session_state
st.set_page_config(page_title="Crypto Bot Dashboard", layout="wide")

st.title("🚀 Crypto Bot — Dashboard")

# Selector de fuente: Live vs. Simulación
sym_key = settings.symbol.replace("/", "")
sim_base = SIMULATIONS_DIR / f"{sym_key}_{settings.timeframe}"
source = st.sidebar.selectbox("Fuente de datos", ["Live (en vivo)", "Última simulación"], index=0)
selected_sim_run = None
if source != "Live (en vivo)":
    runs = []
    if sim_base.exists():
        runs = sorted([p.name for p in sim_base.iterdir() if p.is_dir()])
    if runs:
        selected_sim_run = st.sidebar.selectbox("Selecciona ejecución", runs, index=len(runs) - 1)
    else:
        st.sidebar.info("No hay simulaciones disponibles todavía.")

def get_metrics_dataframes():
    """Devuelve (trades_df, equity_df) según la fuente seleccionada.

    - Live: si METRICS_BASE_URL está definida, lee desde URL:
        {METRICS_BASE_URL}/trades.csv y /equity.csv
      Si no, intenta local en data/metrics/*
    - Simulación: lee local desde la carpeta de la ejecución.
    """
    if source == "Live (en vivo)":
        trades_url = f"{METRICS_BASE_URL.rstrip('/')}/trades.csv" if METRICS_BASE_URL else None
        equity_url = f"{METRICS_BASE_URL.rstrip('/')}/equity.csv" if METRICS_BASE_URL else None
        tdf = _read_csv_local_or_url(TRADES_CSV, trades_url)
        edf = _read_csv_local_or_url(EQUITY_CSV, equity_url)
        return tdf, edf

    # Simulación
    if selected_sim_run:
        base = sim_base / selected_sim_run
        tdf = _read_csv_local_or_url(base / "trades.csv", None)
        edf = _read_csv_local_or_url(base / "equity.csv", None)
        return tdf, edf

    # Fallback
    tdf = _read_csv_local_or_url(TRADES_CSV, None)
    edf = _read_csv_local_or_url(EQUITY_CSV, None)
    return tdf, edf

def compute_kpis_df(tdf: Optional[pd.DataFrame], edf: Optional[pd.DataFrame]):
    """Calcula KPIs del run actual usando equity y trades DataFrames.

    Detecta el inicio del run buscando el último bloque donde la equity
    esté en el rango de un run con capital pequeño (basado en settings.initial_cash).
    """
    if edf is None or edf.empty or "timestamp" not in edf.columns or "equity" not in edf.columns:
        return None
    edf["timestamp"] = pd.to_datetime(edf["timestamp"], utc=True)

    initial_cash = float(getattr(settings, "initial_cash", 0.0) or 0.0)
    # Heurística: si initial_cash está definido (>0), tomamos el último bloque con equity <= initial_cash*2
    # para aislar el run en curso (p.ej. $10). Si no está definido, usamos todo el archivo.
    if initial_cash > 0:
        thr = initial_cash * 2.0
        mask_small = edf["equity"] <= thr
        if mask_small.any():
            last_small_idx = mask_small[mask_small].index.max()
            start_idx = last_small_idx
            while start_idx > 0 and edf.loc[start_idx - 1, "equity"] <= thr:
                start_idx -= 1
            run_edf = edf.iloc[start_idx:].copy()
        else:
            run_edf = edf.copy()
    else:
        run_edf = edf.copy()

    initial_equity = float(run_edf["equity"].iloc[0])
    current_equity = float(run_edf["equity"].iloc[-1])
    pnl_abs = current_equity - initial_equity
    pnl_pct = (pnl_abs / initial_equity) * 100 if initial_equity else 0.0

    cummax = run_edf["equity"].cummax()
    drawdowns = (run_edf["equity"] / cummax) - 1.0
    max_dd_pct = float(drawdowns.min() * 100.0)

    run_start_ts = run_edf["timestamp"].iloc[0]

    buys = sells = exits = total_trades = closed_trades = 0
    win_rate_pct = None
    avg_pnl_pct = None
    best_trade_pct = None
    worst_trade_pct = None
    avg_holding_minutes = None

    if tdf is not None and not tdf.empty and "timestamp" in tdf.columns and "side" in tdf.columns:
        tdf["timestamp"] = pd.to_datetime(tdf["timestamp"], utc=True)
        tdf_run = tdf[tdf["timestamp"] >= run_start_ts].copy()
        total_trades = len(tdf_run)
        buys = int((tdf_run["side"] == "buy").sum())
        sells = int((tdf_run["side"] == "sell").sum())
        exits = int((tdf_run["side"] == "risk-exit").sum())
        closing = tdf_run[tdf_run["side"].isin(["sell", "risk-exit"])].copy()
        if not closing.empty:
            if "pnl_pct" in closing.columns:
                closing["pnl_pct"] = pd.to_numeric(closing["pnl_pct"], errors="coerce")
                wins = int((closing["pnl_pct"] > 0).sum())
                closed_trades = len(closing)
                win_rate_pct = (wins / closed_trades * 100.0) if closed_trades else None
                if closing["pnl_pct"].notna().any():
                    avg_pnl_pct = float(closing["pnl_pct"].mean())
                    best_trade_pct = float(closing["pnl_pct"].max())
                    worst_trade_pct = float(closing["pnl_pct"].min())

            # Tiempo de holding: para cada cierre, buscamos el BUY más reciente anterior
            # Asumimos un modelo simple de posición única (buy -> sell/exit)
            if not tdf_run.empty:
                buy_times = tdf_run[tdf_run["side"] == "buy"]["timestamp"].sort_values().tolist()
                if buy_times:
                    hold_durations = []
                    for ts_close in closing["timestamp"].sort_values().tolist():
                        # último buy anterior al cierre
                        prev_buys = [bt for bt in buy_times if bt <= ts_close]
                        if prev_buys:
                            last_buy = prev_buys[-1]
                            delta = (ts_close - last_buy).total_seconds() / 60.0
                            if delta >= 0:
                                hold_durations.append(delta)
                    if hold_durations:
                        avg_holding_minutes = float(sum(hold_durations) / len(hold_durations))

    return {
        "initial_equity": initial_equity,
        "current_equity": current_equity,
        "pnl_abs": pnl_abs,
        "pnl_pct": pnl_pct,
        "max_drawdown_pct": max_dd_pct,
        "total_trades": total_trades,
        "buys": buys,
        "sells": sells,
        "risk_exits": exits,
        "closed_trades": closed_trades,
        "win_rate_pct": win_rate_pct,
        "avg_pnl_pct": avg_pnl_pct,
        "best_trade_pct": best_trade_pct,
        "worst_trade_pct": worst_trade_pct,
        "avg_holding_minutes": avg_holding_minutes,
    }

col1, col2 = st.columns([2, 1])
with col2:
    st.subheader("Estado & Parámetros")
    st.write(f"Exchange: {settings.exchange_id}")
    st.write(f"Símbolo: {settings.symbol}")
    st.write(f"Timeframe: {settings.timeframe}")
    st.write(f"AI_USE: {settings.ai_use}")
    st.write(f"AI_PROBA_BUY: {settings.ai_proba_buy:.2f}")
    st.write(f"AI_PROBA_SELL: {settings.ai_proba_sell:.2f}")
    st.write(
        f"Riesgo — SL: {settings.risk_stop_loss_pct:.2%}, TP: {settings.risk_take_profit_pct:.2%}, "
        f"TR: {settings.risk_trailing_stop_pct:.2%}\nMaxDailyLoss: {settings.risk_max_daily_loss_pct:.2%}, "
        f"MaxConsecLosses: {settings.risk_max_consecutive_losses}, KillDD: {settings.risk_kill_switch_drawdown_pct:.2%}"
    )

    st.subheader("Risk state")
    if STATE_JSON.exists():
        try:
            state = json.loads(STATE_JSON.read_text(encoding="utf-8"))
            st.json(state)
        except Exception as e:
            st.warning(f"No se pudo leer state.json: {e}")
    else:
        st.info("Sin estado persistido aún.")

    # Diagnóstico rápido de fuentes remotas
    with st.expander("Diagnóstico remoto (Cloud)"):
        st.caption("Usa esto en Streamlit Cloud para verificar que las URLs remotas sean accesibles.")
        st.write(f"METRICS_BASE_URL: {METRICS_BASE_URL or 'NO DEFINIDO'}")
        st.write(f"PRICE_CSV_URL: {PRICE_CSV_URL or 'NO DEFINIDO'}")
        if st.button("Probar endpoints remotos", type="secondary"):
            try:
                results = []
                if METRICS_BASE_URL:
                    try:
                        url_eq = f"{METRICS_BASE_URL.rstrip('/')}/equity.csv"
                        df_eq = pd.read_csv(url_eq, nrows=5)
                        results.append(f"equity.csv OK — filas={len(df_eq)}")
                    except Exception as e:
                        results.append(f"equity.csv ERROR — {e}")
                    try:
                        url_tr = f"{METRICS_BASE_URL.rstrip('/')}/trades.csv"
                        df_tr = pd.read_csv(url_tr, nrows=5)
                        results.append(f"trades.csv OK — filas={len(df_tr)}")
                    except Exception as e:
                        results.append(f"trades.csv ERROR — {e}")
                else:
                    results.append("METRICS_BASE_URL no definido")

                if PRICE_CSV_URL:
                    try:
                        df_px = pd.read_csv(PRICE_CSV_URL, nrows=5)
                        results.append(f"PRICE_CSV_URL OK — filas={len(df_px)}")
                    except Exception as e:
                        results.append(f"PRICE_CSV_URL ERROR — {e}")
                else:
                    results.append("PRICE_CSV_URL no definido (opcional)")

                for line in results:
                    if "OK" in line:
                        st.success(line)
                    else:
                        st.error(line)
            except Exception as e:
                st.error(f"Fallo en diagnóstico: {e}")

    # KPIs del run actual
    st.subheader("KPIs del run actual")
    tdf, edf = get_metrics_dataframes()
    try:
        kpis = compute_kpis_df(tdf, edf)
        if not kpis:
            st.info("Sin KPIs disponibles todavía.")
        else:
            st.write(f"Equity inicial: ${kpis['initial_equity']:.4f}")
            st.write(f"Equity actual: ${kpis['current_equity']:.4f}")
            st.write(f"PnL: ${kpis['pnl_abs']:.4f} ({kpis['pnl_pct']:.3f}%)")
            st.write(f"Máx. drawdown: {kpis['max_drawdown_pct']:.3f}%")
            st.write(f"Trades totales: {kpis['total_trades']} | BUY: {kpis['buys']} | SELL: {kpis['sells']} | EXITS: {kpis['risk_exits']}")
            st.write(f"Cerradas: {kpis['closed_trades']} | Win rate: {kpis['win_rate_pct']:.2f}%" if kpis['win_rate_pct'] is not None else "Cerradas: 0 | Win rate: N/A")
            if kpis["avg_pnl_pct"] is not None:
                st.write(f"PnL medio (cerradas): {kpis['avg_pnl_pct']:.3f}%")
            if kpis["best_trade_pct"] is not None and kpis["worst_trade_pct"] is not None:
                st.write(f"Mejor trade: {kpis['best_trade_pct']:.3f}% | Peor trade: {kpis['worst_trade_pct']:.3f}%")
            if kpis["avg_holding_minutes"] is not None:
                st.write(f"Tiempo medio en posición: {kpis['avg_holding_minutes']:.1f} min")
    except Exception as e:
        st.warning(f"No se pudieron calcular KPIs: {e}")

with col1:
    st.subheader("Precio y señales")
    price_df = load_price_df()
    fig, ax = plt.subplots(figsize=(10, 4))
    if not price_df.empty and "close" in price_df.columns:
        price_df["close"].plot(ax=ax, color="steelblue", label="Close")
        # Plot trades
        tdf, _ = get_metrics_dataframes()
        if tdf is not None and not tdf.empty:
            try:
                tdf["timestamp"] = pd.to_datetime(tdf["timestamp"], utc=True)
                buys = tdf[tdf["side"] == "buy"]
                sells = tdf[tdf["side"] == "sell"]
                rexits = tdf[tdf["side"] == "risk-exit"]
                if not buys.empty:
                    ax.scatter(buys["timestamp"], buys["price"], marker="^", color="green", label="BUY")
                if not sells.empty:
                    ax.scatter(sells["timestamp"], sells["price"], marker="v", color="red", label="SELL")
                if not rexits.empty:
                    ax.scatter(rexits["timestamp"], rexits["price"], marker="x", color="orange", label="RISK-EXIT")
            except Exception:
                pass
        ax.legend(loc="best")
        ax.set_ylabel("Precio")
    else:
        ax.text(0.5, 0.5, "Sin datos de precio", ha="center", va="center")
    st.pyplot(fig)

    # Tabla de últimas operaciones (10 más recientes)
    tdf, _ = get_metrics_dataframes()
    if tdf is not None:
        try:
            if not tdf.empty:
                tdf["timestamp"] = pd.to_datetime(tdf["timestamp"], utc=True)
                tdf = tdf.sort_values("timestamp", ascending=False)
                cols = [c for c in ["timestamp", "side", "price", "qty", "pnl_pct"] if c in tdf.columns]
                st.subheader("Últimas 10 operaciones")
                st.dataframe(tdf[cols].head(10), use_container_width=True)

                # Tabla de cerradas (SELL / RISK-EXIT) — últimas 10
                st.subheader("Cerradas (últimas 10)")
                closed = tdf[tdf["side"].isin(["sell", "risk-exit"])].copy()
                if not closed.empty:
                    if "pnl_pct" in closed.columns:
                        closed["pnl_pct"] = pd.to_numeric(closed["pnl_pct"], errors="coerce")
                    closed = closed.sort_values("timestamp", ascending=False)
                    cols_c = [c for c in ["timestamp", "side", "price", "qty", "pnl_pct"] if c in closed.columns]
                    st.dataframe(closed[cols_c].head(10), use_container_width=True)
                else:
                    st.info("Aún no hay operaciones cerradas.")

                # Histograma de PnL (cerradas)
                if "pnl_pct" in closed.columns and closed["pnl_pct"].notna().any():
                    figh, axh = plt.subplots(figsize=(6, 3))
                    closed["pnl_pct"].dropna().plot(kind="hist", bins=20, color="slateblue", edgecolor="white", ax=axh)
                    axh.set_title("Distribución PnL (%) de cerradas")
                    axh.set_xlabel("PnL %")
                    axh.grid(True, alpha=0.2)
                    st.pyplot(figh)
        except Exception as e:
            st.warning(f"No se pudo mostrar la tabla de operaciones: {e}")

st.subheader("Equity")
fig2, ax2 = plt.subplots(figsize=(10, 3))
_, edf = get_metrics_dataframes()
if edf is not None and not edf.empty:
    try:
        edf["timestamp"] = pd.to_datetime(edf["timestamp"], utc=True)
        edf.set_index("timestamp", inplace=True)
        edf["equity"].plot(ax=ax2, color="purple")
        ax2.set_ylabel("Equity")
        # Monto actual
        if not edf.empty:
            st.info(f"Monto actual (equity): ${float(edf['equity'].iloc[-1]):.2f}")
    except Exception:
        ax2.text(0.5, 0.5, "Error leyendo equity.csv", ha="center", va="center")
else:
    ax2.text(0.5, 0.5, "Sin métricas aún (equity.csv)", ha="center", va="center")
st.pyplot(fig2)

st.caption("Pulsa el botón para actualizar los datos")
if st.sidebar.button("Actualizar ahora"):
    st.rerun()

# Aviso de fuente
if source == "Live (en vivo)" and METRICS_BASE_URL:
    st.caption(f"Leyendo métricas en vivo desde: {METRICS_BASE_URL}")
