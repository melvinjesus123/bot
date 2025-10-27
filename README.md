### Elegir algoritmo de IA (RF o XGBoost)

Puedes seleccionar el algoritmo desde el `.env`:

```
AI_ALGO=rf   # opciones: rf (RandomForest), xgb (XGBoost)
```

Entrenar y afinar usarán ese algoritmo:
- Entrenar: tarea "Train model" o `python scripts/train_model.py`
- Afinar: tarea "Tune model" o `python scripts/tune_model.py`

Notas:
- XGBoost requiere haber instalado la dependencia (`xgboost`) — ya añadida a requirements.txt.
- El motor cargará `models/model_<SYMBOL>_<TF>.joblib` y usará `AI_PROBA_BUY/SELL` para decidir.

## Ejecutarlo 24/7 aunque apagues tu PC (Docker en un VPS)

La forma más sencilla de dejar el bot corriendo 24/7 es desplegarlo en un servidor (VPS) con Docker y docker-compose. Este repo ya trae `docker/Dockerfile` y `docker/docker-compose.yml` con tres servicios: ingest (descarga de datos continua), engine (motor de trading, paper o live según `.env`) y dashboard (Streamlit).

Pasos (Ubuntu 22.04/24.04 típico):

1) Instalar Docker y Compose

   ```bash
   # Como root o con sudo
   apt-get update
   apt-get install -y ca-certificates curl gnupg
   install -m 0755 -d /etc/apt/keyrings
   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
   echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" > /etc/apt/sources.list.d/docker.list
   apt-get update
   apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
   usermod -aG docker $USER  # cierra sesión y vuelve a entrar para aplicar
   ```

2) Clonar el repo y preparar `.env`

   ```bash
   git clone <TU_REPO> bot
   cd bot
   cp configs/example_config.yaml configs/config.yaml  # si te aplica
   # Copia también tu .env local o crea uno nuevo (NO subas este archivo al repo público)
   nano .env
   ```

   Variables importantes en `.env`:
   - PAPER_TRADING=true/false (true para paper, false para real)
   - EXCHANGE_ID, API_KEY, API_SECRET, API_PASSPHRASE (si aplica)
   - INITIAL_CASH, ENGINE_RUN_SECONDS (ponlo vacío o grande si quieres que no pare)
   - LIVE_MAX_TRADE_USD, LIVE_MIN_NOTIONAL_USD (guardas en real)

3) Levantar los servicios

   ```bash
   cd docker
   docker compose up -d --build
   ```

   - Ingest: corre en bucle descargando OHLCV y guardando en `data/`
   - Engine: ejecuta el motor (paper/live según `.env`)
   - Dashboard: expuesto en el puerto 8501 (http://<IP_DEL_VPS>:8501)

4) Logs y control

   ```bash
   docker compose ps
   docker compose logs -f engine
   docker compose logs -f ingest
   docker compose logs -f dashboard
   docker compose restart engine
   docker compose down    # para parar todo
   ```

5) Seguridad y puertos

   - Abre el puerto 8501 solo si quieres ver el dashboard desde fuera (configura firewall de tu VPS).
   - El archivo `.env` contiene credenciales: mantenlo fuera de git y con permisos adecuados.

Alternativas sin VPS propio:
- Render, Railway, Fly.io o similar: sube la imagen Docker y configura las variables de entorno desde el panel.
- GitHub Actions + GHCR: build/push de la imagen a un registro y despliegue automático en tu VPS.

Con este enfoque el bot seguirá corriendo aunque apagues tu computadora local.
# Bot de Trading Cripto (24/7) — Base de Proyecto

Este repositorio es un punto de partida para un bot de trading de criptomonedas que ingiere datos 24/7, permite backtesting y soporta paper trading de forma segura antes de operar en real.

> Aviso importante: El trading conlleva riesgos significativos. No hay garantías de rentabilidad ni de "duplicar ganancias". Usa bajo tu propio riesgo y prueba exhaustivamente en paper trading.

## Requisitos rápidos
- Python 3.11+
- VS Code con extensiones: Python, Pylance, Jupyter, Docker, YAML, GitLens

## Configuración de secretos (imprescindible)
1. Copia el archivo `.env.example` a `.env` y completa tus credenciales del exchange (si vas a probar en real más adelante). Por defecto, `PAPER_TRADING=true` para operar en modo simulado.
2. Variables principales del `.env`:
   - `EXCHANGE_ID` (ej. `binance`)
   - `API_KEY`, `API_SECRET`, `API_PASSPHRASE` (si aplica)
   - `SYMBOL` (ej. `BTC/USDT`), `TIMEFRAME` (ej. `1h`)
   - `DB_URL` (por defecto `sqlite:///data/market.db`)
   - Parámetros de riesgo: `RISK_MAX_POSITION_SIZE`, `RISK_STOP_LOSS_PCT`, `RISK_TAKE_PROFIT_PCT`

Nunca subas tu `.env` a un repositorio público.

## Instalación
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Uso rápido

1) Demo de entorno (no trading real):
```powershell
python .\scripts\run_demo.py
```

2) Ingesta OHLCV una vez (crea CSV en `data/` y guarda en SQLite si `DB_URL` es sqlite):
```powershell
python .\src\ingest\ccxt_ingest.py
```

3) Backtest SMA crossover sobre el CSV descargado:
```powershell
python .\scripts\run_backtest.py
```

4) Motor de paper trading continuo (24/7) con decisiones por SMA crossover:
- Opción A (Python): crea un pequeño lanzador o usa el módulo directamente (ver `src/trader/engine.py`).
- Opción B (Docker):
```powershell
docker compose -f .\docker\docker-compose.yml up --build
```

## IA: entrenamiento y uso

1) Entrenar el modelo (RandomForest con features sencillas: retornos, SMA ratio, RSI):
```powershell
python .\scripts\train_model.py
```
Esto generará un archivo en `models/model_<SYMBOL>_<TIMEFRAME>.joblib`.

2) Activar IA en el motor de paper trading: el engine detecta el modelo automáticamente y combina la probabilidad de subida con la señal SMA para decidir BUY/SELL/HOLD.

3) Ajustar umbrales en `.env`:
```
AI_USE=true
AI_PROBA_BUY=0.55
AI_PROBA_SELL=0.45
```
El motor lee estos valores desde `src/config/settings.py`.

### Auto-reentrenamiento (opcional)

Puedes automatizar el reentrenamiento del modelo para que incorpore datos nuevos de forma periódica:

- Script: `scripts/auto_retrain.py` (entrena cada 24h por defecto y hace tuning cada 7 días)
- Tarea VS Code: "Auto retrain (daily)" (en background)
- Variables de entorno opcionales:
   - `RETRAIN_INTERVAL_HOURS` (por defecto 24)
   - `TUNE_EVERY_N_DAYS` (por defecto 7)

El motor recarga `models/model_<SYMBOL>_<TIMEFRAME>.joblib` automáticamente en cada ciclo, por lo que no es necesario reiniciarlo al generar un nuevo modelo.

## Afinado (mejores parámetros y umbrales)

1) Tuning automático de hiperparámetros (RandomForest, CV temporal):
```powershell
python .\scripts\tune_model.py
```
Esto generará `models/tuning_<SYMBOL>_<TIMEFRAME>.json` con resultados y reentrenará el modelo óptimo.

2) Entrenamiento con métricas y umbral recomendado:
```powershell
python .\scripts\train_model.py
```
Se guardará `models/metrics_<SYMBOL>_<TIMEFRAME>.json` con AUC CV y `thr_recommended`.

3) Gestión de riesgo en el motor:
- Usa `RISK_STOP_LOSS_PCT` y `RISK_TAKE_PROFIT_PCT` (ver `.env`).
- El engine verifica SL/TP antes de tomar nuevas decisiones.

## Próximos pasos
- Ajustar parámetros de estrategias y riesgo.
- Añadir más indicadores/estrategias y pruebas.
- Integración CI (ya incluida) y Docker listos para iterar.

## Seguridad y riesgos
- Mantén `PAPER_TRADING=true` hasta validar completamente tu estrategia.
- Configura límites de tamaño de posición y stop-loss/take-profit por defecto.
- Considera un exchange de prueba/sandbox cuando esté disponible.
 - Consulta la guía de puesta en marcha segura en `docs/SAFETY.md`.

## Mitigación de errores y pérdidas (APIs y controles)
- Reintentos con backoff (tenacity) para llamadas a exchange (ingesta).
- Gestión de riesgo en el engine: SL, TP y Trailing Stop (RISK_STOP_LOSS_PCT, RISK_TAKE_PROFIT_PCT, RISK_TRAILING_STOP_PCT).
- Kill-switch configurable: pérdida diaria máxima, drawdown máximo y pérdidas consecutivas.
- Notificaciones (opcional): Telegram (TELEGRAM_BOT_TOKEN/CHAT_ID) y Slack (SLACK_WEBHOOK_URL).
- Monitorización (opcional): Sentry (SENTRY_DSN) para capturar excepciones.

## Licencia
MIT
