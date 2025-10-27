"""Demo: carga configuración y muestra parámetros clave.

Este script no ejecuta trading real. Sirve para validar que la configuración
por .env funciona y que el entorno está listo.
"""
from __future__ import annotations

from pathlib import Path

from src.config.settings import settings
from src.utils.logging_setup import setup_logging


def main() -> None:
    logger = setup_logging(settings.log_level)

    env_exists = Path(".env").exists()
    logger.info(".env presente: %s", env_exists)
    logger.info(
        "Exchange=%s | Symbol=%s | Timeframe=%s | PaperTrading=%s",
        settings.exchange_id,
        settings.symbol,
        settings.timeframe,
        settings.paper_trading,
    )
    logger.info("DB_URL=%s", settings.db_url)
    logger.info("Demo OK — no se ejecutó trading.")


if __name__ == "__main__":
    main()
