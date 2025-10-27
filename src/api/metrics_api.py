from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse

DATA_DIR = Path("data")
METRICS_DIR = DATA_DIR / "metrics"

app = FastAPI(title="Bot Metrics API", version="1.0.0")

# CORS: permitir orígenes de Streamlit Cloud y otros
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # puedes restringirlo a dominios concretos
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/health", response_class=PlainTextResponse)
def health() -> str:
    return "ok"


@app.get("/metrics/trades.csv")
def get_trades_csv() -> FileResponse:
    path = METRICS_DIR / "trades.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="trades.csv no encontrado")
    return FileResponse(path, media_type="text/csv", filename="trades.csv")


@app.get("/metrics/equity.csv")
def get_equity_csv() -> FileResponse:
    path = METRICS_DIR / "equity.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="equity.csv no encontrado")
    return FileResponse(path, media_type="text/csv", filename="equity.csv")


@app.get("/price/{filename}")
def get_price_csv(filename: str) -> FileResponse:
    # Seguridad básica: solo permitir .csv sin subdirectorios
    if "/" in filename or ".." in filename or not filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="nombre de archivo inválido")
    path = DATA_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{filename} no encontrado")
    return FileResponse(path, media_type="text/csv", filename=filename)
