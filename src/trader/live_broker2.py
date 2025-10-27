from __future__ import annotations

import math
from dataclasses import dataclass

import ccxt  # type: ignore

from src.config.settings import settings


@dataclass
class Position:
    qty: float = 0.0
    entry_price: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.qty > 0


class LiveBroker:
    """Broker real basado en ccxt (órdenes de mercado).

    - Usa un tope por operación en USD (LIVE_MAX_TRADE_USD) y comprueba un mínimo notional.
    - Ajusta la cantidad a la precisión del símbolo.
    - equity() se calcula con balances de base y quote: equity = base*precio + quote.
    """

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.base, self.quote = symbol.split("/")
        ex_id = settings.exchange_id
        api_key = settings.api_key
        secret = settings.api_secret
        password = settings.api_passphrase

        exchange_class = getattr(ccxt, ex_id)
        opts = {
            "enableRateLimit": True,
            "apiKey": api_key,
            "secret": secret,
        }
        if password:
            opts["password"] = password
        self.exchange = exchange_class(opts)
        self.pos = Position()

    def _amount_to_precision(self, amount: float) -> float:
        try:
            return float(self.exchange.amount_to_precision(self.symbol, amount))
        except Exception:
            # fallback a 6 decimales
            return math.floor(amount * 1e6) / 1e6

    def _fetch_balances(self) -> dict:
        return self.exchange.fetch_balance()

    def value(self, mark_price: float) -> float:
        bal = self._fetch_balances()
        base_total = float(bal.get(self.base, {}).get("total", 0.0))
        quote_total = float(bal.get(self.quote, {}).get("total", 0.0))
        return quote_total + base_total * mark_price

    def buy_all(self, price: float) -> None:
        bal = self._fetch_balances()
        quote_free = float(bal.get(self.quote, {}).get("free", 0.0))
        budget = min(quote_free, settings.live_max_trade_usd)
        if budget < settings.live_min_notional_usd:
            # no cumple el mínimo notional
            return
        amount = budget / price
        amount = self._amount_to_precision(amount)
        if amount <= 0:
            return
        # orden de mercado buy
        self.exchange.create_order(self.symbol, "market", "buy", amount)
        # actualizar posición aproximada
        bal2 = self._fetch_balances()
        self.pos.qty = float(bal2.get(self.base, {}).get("free", 0.0))
        self.pos.entry_price = price

    def sell_all(self, price: float) -> None:
        bal = self._fetch_balances()
        base_free = float(bal.get(self.base, {}).get("free", 0.0))
        amount = self._amount_to_precision(base_free)
        if amount <= 0:
            return
        self.exchange.create_order(self.symbol, "market", "sell", amount)
        # actualizar posición
        self.pos.qty = 0.0
        self.pos.entry_price = 0.0
