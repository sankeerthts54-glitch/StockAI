"""
main.py
-------
FastAPI backend for the AI Stock Market Analysis & Prediction System.
Serves REST API endpoints consumed by the frontend dashboard.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import os
import sys

# Ensure backend package is importable when run from project root
sys.path.insert(0, os.path.dirname(__file__))

from data_generator import generate_stock_data, get_all_tickers, STOCK_CONFIGS
from analytics import analyse_stock
from insights import generate_insight

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Stock Market Analysis API",
    description="Analyses historical stock data and generates AI-driven insights.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Serve frontend static files (works locally and safely skips if running pure serverless)
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
PUBLIC_DIR = os.path.join(os.path.dirname(__file__), "..", "public")
target_dir = FRONTEND_DIR if os.path.exists(FRONTEND_DIR) else (PUBLIC_DIR if os.path.exists(PUBLIC_DIR) else None)

if target_dir and os.path.exists(target_dir):
    app.mount("/static", StaticFiles(directory=target_dir), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        """Serve the main dashboard HTML."""
        index_path = os.path.join(target_dir, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"status": "StockAI API running. Visit frontend."}


@app.get("/api/tickers", summary="List available ticker symbols")
async def list_tickers():
    """Returns all available stock tickers with their company names."""
    return {"tickers": get_all_tickers()}


@app.get("/api/stock/{ticker}", summary="Full analysis for one ticker")
async def get_stock_analysis(ticker: str):
    """
    Returns historical OHLCV data, SMA50, SMA200, RSI,
    ML price forecasts, and AI-generated insights for a given ticker.
    """
    ticker = ticker.upper()
    if ticker not in STOCK_CONFIGS:
        raise HTTPException(
            status_code=404,
            detail=f"Ticker '{ticker}' not found. Available: {list(STOCK_CONFIGS.keys())}",
        )

    # Generate & analyse
    df      = generate_stock_data(ticker, days=365)
    result  = analyse_stock(df, ticker)
    metrics = result["metrics"]

    # Generate NL insights
    insight = generate_insight(
        ticker          = ticker,
        current_price   = metrics["current_price"],
        sma50           = metrics["sma50"],
        sma200          = metrics["sma200"],
        predicted_prices= result["predicted_prices"],
        rsi             = metrics["rsi"],
    )

    result["insight"] = insight
    return result


@app.get("/api/health", summary="Health check")
async def health():
    return {"status": "ok", "version": "1.0.0"}
