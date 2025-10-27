import logging
import sys
from typing import Optional

# Sentry es opcional: importamos de forma segura y solo inicializamos si hay DSN.
try:  # pragma: no cover - import condicional
    from sentry_sdk import Hub as _SentryHub  # type: ignore
    from sentry_sdk import init as _sentry_init  # type: ignore
    from sentry_sdk.integrations.logging import (  # type: ignore
        LoggingIntegration as _SentryLoggingIntegration,
    )
except Exception:  # noqa: BLE001 - si no está instalado o falla, continuamos sin Sentry
    _SentryHub = None  # type: ignore
    _sentry_init = None  # type: ignore
    _SentryLoggingIntegration = None  # type: ignore


def setup_logging(level: Optional[str] = "INFO") -> logging.Logger:
    """Configura logging a stdout con un formato simple.

    Parameters
    ----------
    level: str
        Nivel de logging (DEBUG, INFO, WARNING, ERROR)
    """
    numeric_level = getattr(logging, (level or "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Inicializa Sentry si hay DSN configurado y aún no está inicializado.
    try:
        if _sentry_init is not None and _SentryLoggingIntegration is not None:
            # Import local para evitar posibles ciclos en tiempo de importación.
            from src.config.settings import settings  # lazy import

            if getattr(settings, "sentry_dsn", None):
                already_initialized = False
                try:
                    if _SentryHub is not None and _SentryHub.current.client is not None:
                        already_initialized = True
                except Exception:
                    # Si no podemos comprobar el estado, intentamos inicializar igualmente.
                    already_initialized = False

                if not already_initialized:
                    sentry_logging = _SentryLoggingIntegration(
                        level=logging.INFO,  # registrar breadcrumbs desde INFO
                        event_level=logging.ERROR,  # enviar eventos desde ERROR
                    )
                    _sentry_init(
                        dsn=settings.sentry_dsn,
                        integrations=[sentry_logging],
                        traces_sample_rate=0.0,  # desactivado por defecto (se puede subir si se desea APM)
                        profiles_sample_rate=0.0,
                        send_default_pii=False,
                    )
    except Exception:
        # Nunca romper el arranque de logging por un fallo de Sentry.
        pass

    return logging.getLogger("bot")
