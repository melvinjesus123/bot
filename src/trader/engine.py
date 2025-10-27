# flake8: noqa
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import pandas as pd

from src.config.settings import settings
from src.utils.logging_setup import setup_logging
from src.ingest.ccxt_ingest import fetch_ohlcv_once
from src.backtest.simple_backtester import sma_crossover_signals
from src.utils.notifier import Notifier
from src.trader.risk import (
    load_state,
    save_state,
    start_of_day_reset,
    update_equity_extremes,
    register_trade_result,
    check_kill_switch,
)
from src.ai.features import make_features, split_xy
from src.ai.model import AiModel
from src.trader.metrics import append_trade, append_equity
from src.trader.live_broker2 import LiveBroker


@dataclass
class Position:
    qty: float = 0.0
    entry_price: float = 0.0
    high_price: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.qty > 0


class PaperBroker:
    def __init__(self, fee: float = 0.0005):
        from src.config.settings import settings as _s
        self.cash = float(getattr(_s, "initial_cash", 1000.0))
        self.pos = Position()
        self.fee = fee

    def value(self, mark_price: float) -> float:
        return self.cash + self.pos.qty * mark_price

    def buy_all(self, price: float) -> None:
        if self.pos.is_open:
            return
        qty = (self.cash * (1 - self.fee)) / price
        self.pos = Position(qty=qty, entry_price=price, high_price=price)
        self.cash = 0.0

    def sell_all(self, price: float) -> None:
        if not self.pos.is_open:
            return
        self.cash = self.pos.qty * price * (1 - self.fee)
        self.pos = Position()


def _check_risk_and_exit(logger: logging.Logger, broker: PaperBroker, price: float) -> bool:
    """Aplica stop-loss/take-profit simples respecto al precio de entrada.

    Devuelve True si se cierra la posición.
    """
    if not broker.pos.is_open:
        return False
    sl = settings.risk_stop_loss_pct
    tp = settings.risk_take_profit_pct
    entry = broker.pos.entry_price
    # trailing stop
    if settings.risk_trailing_stop_pct > 0 and broker.pos.high_price:
        trail = broker.pos.high_price * (1 - settings.risk_trailing_stop_pct)
        if price <= trail:
            broker.sell_all(price)
            logger.info("TRAILING STOP activado: exit @ %.2f | equity=%.2f", price, broker.value(price))
            return True
    # stop-loss fijo
    if price <= entry * (1 - sl):
        broker.sell_all(price)
        logger.info("STOP LOSS activado: exit @ %.2f | equity=%.2f", price, broker.value(price))
        return True
    if price >= entry * (1 + tp):
        broker.sell_all(price)
        logger.info("TAKE PROFIT activado: exit @ %.2f | equity=%.2f", price, broker.value(price))
        return True
    return False


def _load_latest_model_path(symbol: str, timeframe: str) -> Path | None:
    from pathlib import Path
    sym = symbol.replace("/", "")
    tf = timeframe
    path = Path("models") / f"model_{sym}_{tf}.joblib"
    return path if path.exists() else None


async def decide_and_trade_once(
    logger: logging.Logger,
    broker: PaperBroker,
    notifier: Optional[Notifier] = None,
    use_ai: Optional[bool] = None,
    proba_buy: Optional[float] = None,
    proba_sell: Optional[float] = None,
) -> None:
    df = await fetch_ohlcv_once(
        settings.exchange_id, settings.symbol, settings.timeframe, limit=200
    )
    # SMA para filtro de tendencia configurable
    sig = sma_crossover_signals(df, fast=settings.ai_sma_fast, slow=settings.ai_sma_slow)
    last = sig.iloc[-1]

    notifier = notifier or Notifier()
    # estado de riesgo
    state = load_state()
    price = float(last["close"]) if pd.notna(last["close"]) else None
    ts_iso = None
    if "datetime" in last and pd.notna(last["datetime"]):
        try:
            ts_iso = pd.to_datetime(last["datetime"]).isoformat()
        except Exception:
            ts_iso = None
    if price is None:
        logger.warning("Precio no disponible en la última vela")
        return
    # actualizar high watermark de posición
    if broker.pos.is_open:
        broker.pos.high_price = max(broker.pos.high_price, price)

    # Configuración IA (usa settings como default si no se pasan overrides)
    _use_ai = settings.ai_use if use_ai is None else use_ai
    _proba_buy = settings.ai_proba_buy if proba_buy is None else proba_buy
    _proba_sell = settings.ai_proba_sell if proba_sell is None else proba_sell

    ai_decision = None
    if _use_ai:
        feats = make_features(df)
        X, _ = split_xy(feats)
        model_path = _load_latest_model_path(settings.symbol, settings.timeframe)
        if model_path:
            model = AiModel()
            model.load(model_path)
            proba_up = float(model.predict_proba_up(X.tail(1))[0])
            logger.info("IA proba_up=%.3f | buy>=%.2f sell<=%.2f", proba_up, _proba_buy, _proba_sell)
            if proba_up >= _proba_buy:
                ai_decision = "buy"
            elif proba_up <= _proba_sell:
                ai_decision = "sell"

    # Gating por tendencia SMA: solo comprar si sma_fast > sma_slow; vender si se pierde la tendencia (opcional)
    trend_ok = True
    if settings.ai_sma_filter_use:
        try:
            sf = float(last.get("sma_fast"))
            ss = float(last.get("sma_slow"))
            if pd.notna(sf) and pd.notna(ss):
                trend_ok = sf > ss
        except Exception:
            trend_ok = True

    if settings.ai_sma_filter_use and ai_decision == "buy" and not trend_ok:
        # Bloquea la compra si no hay tendencia alcista
        ai_decision = None
    if (
        settings.ai_sma_filter_use
        and broker.pos.is_open
        and not trend_ok
        and settings.ai_sma_sell_on_trend_loss
    ):
        # Forzar venta por pérdida de tendencia
        ai_decision = "sell"

    # primero aplica gestión de riesgo si hay posición
    entry_before = broker.pos.entry_price if broker.pos.is_open else None
    if broker.pos.is_open and _check_risk_and_exit(logger, broker, price):
        # registrar resultado del trade usando entry_before (posición ya cerrada)
        if entry_before and entry_before > 0:
            pnl_pct = (price - entry_before) / entry_before
        else:
            pnl_pct = 0.0
        state = register_trade_result(state, pnl_pct)
        msg = f"[RISK EXIT] PnL={pnl_pct*100:.2f}% @ {price:.2f}"
        await notifier.send(msg)
        save_state(state)
        if ts_iso:
            append_trade(ts_iso, "risk-exit", price, 0.0, broker.value(price), pnl_pct)
            append_equity(ts_iso, broker.value(price))
        return

    # decisión combinada: IA prioritaria si existe, si no usa SMA
    if ai_decision == "buy" or (ai_decision is None and last["signal"] == 1 and not broker.pos.is_open):
        qty_before = broker.pos.qty
        broker.buy_all(price)
        logger.info("PAPER BUY @ %.2f | equity=%.2f", price, broker.value(price))
        await notifier.send(f"[BUY] {settings.symbol} @{price:.2f}")
        if ts_iso:
            append_trade(ts_iso, "buy", price, broker.pos.qty, broker.value(price), None)
    elif ai_decision == "sell" or (ai_decision is None and last["signal"] == -1 and broker.pos.is_open):
        # calcular pnl aproximado antes de cerrar
        entry_px = broker.pos.entry_price if broker.pos.entry_price else price
        qty_before = broker.pos.qty
        pnl_pct = (price - entry_px) / entry_px if entry_px else 0.0
        broker.sell_all(price)
        logger.info("PAPER SELL @ %.2f | equity=%.2f", price, broker.value(price))
        state = register_trade_result(state, pnl_pct)
        msg = f"[SELL] {settings.symbol} @{price:.2f} | PnL={pnl_pct*100:.2f}%"
        await notifier.send(msg)
        if ts_iso:
            append_trade(ts_iso, "sell", price, qty_before, broker.value(price), pnl_pct)
    else:
        logger.info("HOLD | equity=%.2f", broker.value(price))

    # actualizar estado y evaluar kill-switch
    equity = broker.value(price)
    state = start_of_day_reset(state, equity)
    state = update_equity_extremes(state, equity)
    kill, reason = check_kill_switch(state, equity)
    save_state(state)
    if ts_iso:
        append_equity(ts_iso, equity)
    if kill:
        await notifier.send(f"[KILL-SWITCH] {reason}. Pausando decisiones.")
        logger.warning("KILL-SWITCH: %s", reason)
        return


async def run_paper_engine_forever(sleep_seconds: int = 60) -> None:
    logger = setup_logging(settings.log_level)
    if settings.paper_trading:
        logger.warning("Motor PAPER trading activo. No se enviarán órdenes reales.")
    else:
        logger.warning("MODO LIVE: se enviarán órdenes reales al exchange. Asegúrate de las credenciales y límites.")
    if settings.engine_run_seconds and settings.engine_run_seconds > 0:
        logger.info(
            "Auto-stop activado: ejecutar durante %.2f horas",
            settings.engine_run_seconds / 3600.0,
        )
    broker = PaperBroker() if settings.paper_trading else LiveBroker(settings.symbol)
    notifier = Notifier()
    import time
    deadline = (
        time.monotonic() + settings.engine_run_seconds
        if (settings.engine_run_seconds and settings.engine_run_seconds > 0)
        else None
    )
    while True:
        try:
            await decide_and_trade_once(logger, broker=broker, notifier=notifier)
        except Exception as e:  # noqa: BLE001
            logger.exception("Error en engine: %s", e)
        await asyncio.sleep(sleep_seconds)
        if deadline and time.monotonic() >= deadline:
            logger.info("Tiempo límite alcanzado; deteniendo motor paper.")
            break


if __name__ == "__main__":
    asyncio.run(run_paper_engine_forever())
