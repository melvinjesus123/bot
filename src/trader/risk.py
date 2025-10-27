from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path

from src.config.settings import settings


STATE_PATH = Path("data/state.json")


@dataclass
class RiskState:
    day: str
    day_start_equity: float
    day_max_equity: float
    day_min_equity: float
    cumulative_max_equity: float
    cumulative_min_equity: float
    consecutive_losses: int


def load_state() -> RiskState:
    if STATE_PATH.exists():
        try:
            data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            return RiskState(**data)
        except Exception:
            pass
    # inicial
    return RiskState(
        day=str(date.today()),
        day_start_equity=1000.0,
        day_max_equity=1000.0,
        day_min_equity=1000.0,
        cumulative_max_equity=1000.0,
        cumulative_min_equity=1000.0,
        consecutive_losses=0,
    )


def save_state(state: RiskState) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2), encoding="utf-8")


def start_of_day_reset(state: RiskState, current_equity: float) -> RiskState:
    today = str(date.today())
    if state.day != today:
        state.day = today
        state.day_start_equity = current_equity
        state.day_max_equity = current_equity
        state.day_min_equity = current_equity
        state.consecutive_losses = 0
    return state


def update_equity_extremes(state: RiskState, equity: float) -> RiskState:
    state.day_max_equity = max(state.day_max_equity, equity)
    state.day_min_equity = min(state.day_min_equity, equity)
    state.cumulative_max_equity = max(state.cumulative_max_equity, equity)
    state.cumulative_min_equity = min(state.cumulative_min_equity, equity)
    return state


def register_trade_result(state: RiskState, pnl_pct: float) -> RiskState:
    if pnl_pct < 0:
        state.consecutive_losses += 1
    else:
        state.consecutive_losses = 0
    return state


def check_kill_switch(state: RiskState, equity: float) -> tuple[bool, str | None]:
    # pérdida diaria
    max_daily_loss = settings.risk_max_daily_loss_pct
    if equity <= state.day_start_equity * (1 - max_daily_loss):
        return True, f"Max daily loss {max_daily_loss*100:.1f}% alcanzado"

    # drawdown acumulado
    dd = 1 - (equity / max(1e-9, state.cumulative_max_equity))
    if dd >= settings.risk_kill_switch_drawdown_pct:
        return True, f"Kill-switch por drawdown del {dd*100:.1f}%"

    # pérdidas consecutivas
    if state.consecutive_losses >= settings.risk_max_consecutive_losses:
        return True, f"Kill-switch por {state.consecutive_losses} pérdidas consecutivas"

    return False, None
