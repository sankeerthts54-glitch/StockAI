"""
data_generator.py
-----------------
Generates realistic mock historical stock market data for demonstration purposes.
Uses geometric Brownian motion to simulate realistic price movements.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Seed for reproducibility
SEED = 42

STOCK_CONFIGS = {
    "AAPL": {
        "name": "Apple Inc.",
        "start_price": 170.0,
        "mu": 0.0003,       # daily drift
        "sigma": 0.015,     # daily volatility
        "avg_volume": 75_000_000,
    },
    "TSLA": {
        "name": "Tesla Inc.",
        "start_price": 220.0,
        "mu": 0.0005,
        "sigma": 0.030,
        "avg_volume": 120_000_000,
    },
    "GOOGL": {
        "name": "Alphabet Inc.",
        "start_price": 140.0,
        "mu": 0.0002,
        "sigma": 0.012,
        "avg_volume": 25_000_000,
    },
    "MSFT": {
        "name": "Microsoft Corp.",
        "start_price": 380.0,
        "mu": 0.0004,
        "sigma": 0.013,
        "avg_volume": 22_000_000,
    },
}


def generate_stock_data(ticker: str, days: int = 365) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data for a given ticker using geometric Brownian motion.

    Args:
        ticker: Stock ticker symbol (must exist in STOCK_CONFIGS).
        days:   Number of trading days to simulate.

    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Volume.
    """
    if ticker not in STOCK_CONFIGS:
        raise ValueError(f"Unknown ticker '{ticker}'. Choose from {list(STOCK_CONFIGS.keys())}.")

    cfg = STOCK_CONFIGS[ticker]
    rng = np.random.default_rng(SEED + hash(ticker) % 1000)

    # Simulate closing prices via GBM
    daily_returns = rng.normal(cfg["mu"], cfg["sigma"], days)
    close_prices = cfg["start_price"] * np.exp(np.cumsum(daily_returns))

    # Derive OHLV from closing prices
    daily_range = close_prices * rng.uniform(0.005, 0.025, days)
    open_prices  = close_prices - rng.uniform(-0.5, 0.5, days) * daily_range
    high_prices  = np.maximum(close_prices, open_prices) + rng.uniform(0, 0.5, days) * daily_range
    low_prices   = np.minimum(close_prices, open_prices) - rng.uniform(0, 0.5, days) * daily_range
    volumes      = (cfg["avg_volume"] * rng.uniform(0.5, 1.8, days)).astype(int)

    # Build date index (business days ending today)
    end_date   = datetime.now().date()
    start_date = end_date - timedelta(days=days * 1.5)  # extra buffer for weekends
    bdays      = pd.bdate_range(start=start_date, end=end_date)[-days:]

    df = pd.DataFrame(
        {
            "Date":   bdays,
            "Open":   np.round(open_prices, 2),
            "High":   np.round(high_prices, 2),
            "Low":    np.round(low_prices, 2),
            "Close":  np.round(close_prices, 2),
            "Volume": volumes,
        }
    )
    df["Date"] = df["Date"].astype(str)
    return df


def get_all_tickers() -> list[dict]:
    """Return ticker metadata for the UI dropdown."""
    return [
        {"ticker": t, "name": cfg["name"]}
        for t, cfg in STOCK_CONFIGS.items()
    ]
