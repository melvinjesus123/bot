from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd

from src.config.settings import settings
from tenacity import retry, stop_after_attempt, wait_exponential
from src.utils.logging_setup import setup_logging


async def _create_exchange(exchange_id: str):
    import ccxt.async_support as ccxt  # type: ignore

    if exchange_id not in ccxt.exchanges:
        raise ValueError(f"Exchange no soportado: {exchange_id}")

    klass = getattr(ccxt, exchange_id)
    exchange = klass(
        {
            "apiKey": settings.api_key,
            "secret": settings.api_secret,
            "password": settings.api_passphrase,
            "enableRateLimit": True,
        }
    )
    return exchange


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=30))
async def fetch_ohlcv_once(
    exchange_id: str,
    symbol: str,
    timeframe: str = "1h",
    since_ms: Optional[int] = None,
    limit: int = 500,
) -> pd.DataFrame:
    """Descarga OHLCV una vez y devuelve un DataFrame con columnas estándar.

    Columns: timestamp, open, high, low, close, volume
    """
    exchange = await _create_exchange(exchange_id)
    try:
        await exchange.load_markets()
        raw: List[List[Any]] = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
        df = pd.DataFrame(
            raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        # Normaliza timestamp a UTC ISO
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df
    finally:
        await exchange.close()


def _data_paths(symbol: str, timeframe: str) -> tuple[Path, Path]:
    safe_symbol = symbol.replace("/", "")
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / f"{safe_symbol}_{timeframe}.csv"
    if settings.db_url.startswith("sqlite:///"):
        sqlite_file = settings.db_url.replace("sqlite:///", "")
        sqlite_path = Path(sqlite_file)
    else:
        sqlite_path = Path("data/market.db")
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    return csv_path, sqlite_path


def save_to_csv(df: pd.DataFrame, symbol: str, timeframe: str) -> Path:
    csv_path, _ = _data_paths(symbol, timeframe)
    exists = csv_path.exists()
    if exists:
        # evita duplicados por timestamp
        old = pd.read_csv(csv_path)
        df_all = pd.concat([old, df], ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    else:
        df_all = df.copy()
    df_all.to_csv(csv_path, index=False)
    return csv_path


def save_to_sqlite(df: pd.DataFrame, table_name: str = "ohlcv") -> None:
    if not settings.db_url.startswith("sqlite"):
        return  # en esta base estable, solo soportamos sqlite por defecto
    from sqlalchemy import create_engine

    engine = create_engine(settings.db_url)
    df.to_sql(table_name, con=engine, if_exists="append", index=False)


async def run_once() -> None:
    logger = setup_logging(settings.log_level)
    logger.info("Descargando OHLCV: %s %s %s", settings.exchange_id, settings.symbol, settings.timeframe)
    df = await fetch_ohlcv_once(settings.exchange_id, settings.symbol, settings.timeframe)
    csv_path = save_to_csv(df, settings.symbol, settings.timeframe)
    save_to_sqlite(df)
    logger.info("Guardado CSV en %s | filas=%d", csv_path, len(df))


async def run_forever(sleep_seconds: int = 60) -> None:
    """Loop simple 24/7: descarga periódicamente y guarda.

    Nota: Ajusta sleep_seconds para alinearlo con tu timeframe.
    """
    logger = setup_logging(settings.log_level)
    while True:
        try:
            await run_once()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Fallo en ingestión: %s", exc)
        await asyncio.sleep(sleep_seconds)


if __name__ == "__main__":
    # Ejecuta una sola vez para test rápido
    asyncio.run(run_once())
