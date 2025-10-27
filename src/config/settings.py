from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuración del bot cargada desde variables de entorno (.env).

    Notas:
    - Por defecto usa sqlite en data/market.db
    - PAPER_TRADING=true por seguridad
    - Ajusta los parámetros de riesgo antes de operar en real
    """

    # Exchange y credenciales
    exchange_id: str = Field(default="binance", alias="EXCHANGE_ID")
    api_key: str | None = Field(default=None, alias="API_KEY")
    api_secret: str | None = Field(default=None, alias="API_SECRET")
    api_passphrase: str | None = Field(default=None, alias="API_PASSPHRASE")

    # Mercado y timeframe
    symbol: str = Field(default="BTC/USDT", alias="SYMBOL")
    timeframe: str = Field(default="1h", alias="TIMEFRAME")

    # Base de datos
    db_url: str = Field(default="sqlite:///data/market.db", alias="DB_URL")

    # Modo de operación y riesgo
    paper_trading: bool = Field(default=True, alias="PAPER_TRADING")
    risk_max_position_size: float = Field(default=0.1, alias="RISK_MAX_POSITION_SIZE")
    risk_stop_loss_pct: float = Field(default=0.02, alias="RISK_STOP_LOSS_PCT")
    risk_take_profit_pct: float = Field(default=0.04, alias="RISK_TAKE_PROFIT_PCT")

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # IA
    ai_use: bool = Field(default=True, alias="AI_USE")
    ai_algo: str = Field(default="rf", alias="AI_ALGO")  # 'rf' (RandomForest) | 'xgb' (XGBoost)
    ai_proba_buy: float = Field(default=0.55, alias="AI_PROBA_BUY")
    ai_proba_sell: float = Field(default=0.45, alias="AI_PROBA_SELL")
    # Filtro de tendencia con SMA para gating de IA
    ai_sma_filter_use: bool = Field(default=False, alias="AI_SMA_FILTER_USE")
    ai_sma_fast: int = Field(default=20, alias="AI_SMA_FAST")
    ai_sma_slow: int = Field(default=50, alias="AI_SMA_SLOW")
    ai_sma_sell_on_trend_loss: bool = Field(default=True, alias="AI_SMA_SELL_ON_TREND_LOSS")

    # Capital inicial (paper/simulación) y duración opcional del motor
    initial_cash: float = Field(default=1000.0, alias="INITIAL_CASH")
    engine_run_seconds: int = Field(default=0, alias="ENGINE_RUN_SECONDS")

    # Live trading (reales)
    live_max_trade_usd: float = Field(default=20.0, alias="LIVE_MAX_TRADE_USD")
    live_min_notional_usd: float = Field(default=10.0, alias="LIVE_MIN_NOTIONAL_USD")

    # Notificaciones (opcionales)
    telegram_bot_token: str | None = Field(default=None, alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str | None = Field(default=None, alias="TELEGRAM_CHAT_ID")
    slack_webhook_url: str | None = Field(default=None, alias="SLACK_WEBHOOK_URL")

    # Observabilidad (opcional)
    sentry_dsn: str | None = Field(default=None, alias="SENTRY_DSN")

    # Riesgo avanzado
    risk_trailing_stop_pct: float = Field(default=0.02, alias="RISK_TRAILING_STOP_PCT")
    risk_max_daily_loss_pct: float = Field(default=0.05, alias="RISK_MAX_DAILY_LOSS_PCT")
    risk_max_consecutive_losses: int = Field(default=3, alias="RISK_MAX_CONSECUTIVE_LOSSES")
    risk_kill_switch_drawdown_pct: float = Field(default=0.10, alias="RISK_KILL_SWITCH_DRAWDOWN_PCT")

    # Config de pydantic-settings
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
