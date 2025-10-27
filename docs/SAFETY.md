# Puesta en marcha segura (paper/live)

Antes de operar con dinero real:

1. Paper trading obligatorio
   - Mantén `PAPER_TRADING=true` en `.env` hasta que tengas al menos 4-8 semanas de resultados consistentes.
   - Usa cuentas sandbox si el exchange las ofrece.

2. Gestión de riesgo
   - `RISK_MAX_POSITION_SIZE`: porcentaje máximo del capital en una sola posición (ej. 0.05 = 5%).
   - `RISK_STOP_LOSS_PCT` y `RISK_TAKE_PROFIT_PCT`: define umbrales conservadores y revísalos regularmente.

3. Límites y protecciones
   - Implementa límites diarios de pérdidas y beneficios para pausar el sistema si se alcanzan.
   - Monitoriza logs y alertas (futuros: Slack/Telegram, dashboards).

4. Claves y seguridad
   - Nunca subas `.env` al control de versiones.
   - Usa permisos mínimos: claves con sólo trading (sin retiros), IP allowlist si es posible.

5. Checklist de despliegue
   - Revisa que el exchange y el símbolo sean correctos.
   - Verifica timezone y sincronización de velas/timeframe.
   - Ejecuta un backtest reciente con datos actualizados.
   - Revisa fees, deslizamiento (slippage) y latencia.

6. Descargo de responsabilidad
   - El trading conlleva riesgos significativos. No hay garantías de beneficio.
   - Este repositorio es de propósito educativo; usa bajo tu propio riesgo.
