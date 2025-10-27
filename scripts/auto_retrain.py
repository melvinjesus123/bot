"""Auto-reentrenamiento del modelo IA en ciclo programado.

- Por defecto, entrena cada 24 horas (RETRAIN_INTERVAL_HOURS=24)
- Opcional: ejecutar tuning cada N días (TUNE_EVERY_N_DAYS=7)

El motor de trading recarga automáticamente el archivo de modelo
models/model_<SYMBOL>_<TIMEFRAME>.joblib, por lo que no es necesario
reiniciarlo tras un nuevo entrenamiento.

Uso:
    python scripts/auto_retrain.py

Variables de entorno opcionales:
    RETRAIN_INTERVAL_HOURS=24
    TUNE_EVERY_N_DAYS=7

Nota: Requiere que los datos en data/<SYMBOL>_<TF>.csv estén actualizados
(usa la tarea de ingesta continua).
"""
from __future__ import annotations

import os
import time
import json
from datetime import datetime, timezone
from pathlib import Path

from src.utils.logging_setup import setup_logging
from src.config.settings import settings


DATA_DIR = Path("data")
STATE_PATH = DATA_DIR / "auto_retrain_state.json"
LOCK_PATH = DATA_DIR / "auto_retrain.lock"
RESTART_FLAG_PATH = DATA_DIR / "auto_retrain_restart.flag"


def _load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_schedule_from_envfile() -> tuple[float, int]:
    """Lee RETRAIN_INTERVAL_HOURS y TUNE_EVERY_N_DAYS desde .env si existe.

    - Prioriza valores del archivo .env para permitir hot-reload del intervalo sin reiniciar la tarea.
    - Fallback a variables de entorno del proceso y defaults.
    """
    # valores por defecto / entorno del proceso
    interval_hours = os.getenv("RETRAIN_INTERVAL_HOURS") or "24"
    tune_days = os.getenv("TUNE_EVERY_N_DAYS") or "7"
    try:
        content = Path(".env").read_text(encoding="utf-8")
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k == "RETRAIN_INTERVAL_HOURS":
                interval_hours = v
            elif k == "TUNE_EVERY_N_DAYS":
                tune_days = v
    except Exception:
        # si no hay .env o no se puede leer, usar valores existentes
        pass
    try:
        ih = float(interval_hours)
    except Exception:
        ih = 24.0
    try:
        td = int(tune_days)
    except Exception:
        td = 7
    return ih, td


def _maybe_tune(logger) -> None:
    # lee desde .env para permitir cambios sin reiniciar
    _, tune_every = _read_schedule_from_envfile()
    if tune_every <= 0:
        return
    state = _load_state()
    last_tune_day = state.get("last_tune_day")
    today = datetime.now(timezone.utc).date().isoformat()
    if last_tune_day == today:
        return
    # Si han pasado >= tune_every días desde el último tuning
    last_dt = state.get("last_tune_date_iso")
    should_tune = False
    if not last_dt:
        should_tune = True
    else:
        try:
            last = datetime.fromisoformat(last_dt).date()
            delta = datetime.now(timezone.utc).date() - last
            should_tune = delta.days >= tune_every
        except Exception:
            should_tune = True
    if should_tune:
        try:
            from scripts.tune_model import main as tune_main
            logger.info("[AUTO] Ejecutando tuning de hiperparámetros...")
            tune_main(None, None)
            state["last_tune_day"] = today
            state["last_tune_date_iso"] = datetime.now(timezone.utc).isoformat()
            _save_state(state)
            logger.info("[AUTO] Tuning completado.")
        except Exception as e:  # noqa: BLE001
            logger.exception("[AUTO] Error durante tuning: %s", e)


def _train(logger) -> None:
    try:
        from scripts.train_model import main as train_main
        logger.info("[AUTO] Entrenando modelo (%s)...", getattr(settings, "ai_algo", "rf"))
        train_main(None, None)
        logger.info("[AUTO] Entrenamiento completado.")
    except Exception as e:  # noqa: BLE001
        logger.exception("[AUTO] Error durante entrenamiento: %s", e)


def main() -> None:
    logger = setup_logging(settings.log_level)
    # asegurar carpeta de trabajo
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Gestión de instancia única con lock
    if os.getenv("FORCE_RESTART", "0") == "1" and LOCK_PATH.exists():
        try:
            LOCK_PATH.unlink()
            logger.info("[AUTO] FORCED: lock eliminado para reinicio.")
        except Exception:
            logger.warning("[AUTO] No se pudo eliminar lock forzado.")

    wait_secs = 5.0
    while LOCK_PATH.exists():
        logger.info("[AUTO] Ya hay una instancia corriendo. Esperando %.0f s...", wait_secs)
        time.sleep(wait_secs)
    try:
        LOCK_PATH.write_text(
            json.dumps({
                "pid": os.getpid(),
                "started_iso": datetime.now(timezone.utc).isoformat(),
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        logger.warning("[AUTO] No se pudo escribir lock; continuando igual.")
    ih, _ = _read_schedule_from_envfile()
    logger.info(
        "Auto-retrain activo: cada %.2f horas | SYMBOL=%s | TF=%s | AI_ALGO=%s",
        ih,
        settings.symbol,
        settings.timeframe,
        getattr(settings, "ai_algo", "rf"),
    )
    try:
        while True:
            start = time.monotonic()
            _maybe_tune(logger)
            _train(logger)

            # ¿Se solicitó reinicio? (flag externo)
            if RESTART_FLAG_PATH.exists():
                try:
                    RESTART_FLAG_PATH.unlink()
                except Exception:
                    pass
                logger.info("[AUTO] Reinicio solicitado: saliendo del ciclo para relanzar.")
                break

            # lee el intervalo en cada ciclo desde .env para hot-reload
            ih, _ = _read_schedule_from_envfile()
            interval_seconds = max(3600.0, ih * 3600.0)
            elapsed = time.monotonic() - start
            sleep_for = max(60.0, interval_seconds - elapsed)
            logger.info(
                "Siguiente ciclo en %.2f minutos (intervalo=%.2f h)",
                sleep_for / 60.0,
                ih,
            )
            time.sleep(sleep_for)
    finally:
        try:
            if LOCK_PATH.exists():
                LOCK_PATH.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    main()
