"""
analytics.py
------------
Core analytical engine: computes technical indicators (SMA, RSI) and
trains a Scikit-learn Linear Regression model to forecast future prices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
except ImportError:
    class StandardScaler:  # type: ignore
        def __init__(self):
            self.mean_ = None
            self.scale_ = None

        def fit_transform(self, X):
            X = np.asarray(X, dtype=float)
            self.mean_ = np.mean(X, axis=0)
            self.scale_ = np.std(X, axis=0)
            self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
            return (X - self.mean_) / self.scale_

        def transform(self, X):
            X = np.asarray(X, dtype=float)
            return (X - self.mean_) / self.scale_

        def inverse_transform(self, y):
            y = np.asarray(y, dtype=float)
            return y * self.scale_ + self.mean_

    class LinearRegression:  # type: ignore
        def __init__(self):
            self.coef_ = None

        def fit(self, X, y):
            X_b = np.c_[np.ones((X.shape[0], 1)), X]
            self.coef_, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)
            return self

        def predict(self, X):
            X_b = np.c_[np.ones((X.shape[0], 1)), X]
            return np.dot(X_b, self.coef_)

        def score(self, X, y):
            y_pred = self.predict(X)
            u = ((y - y_pred) ** 2).sum()
            v = ((y - y.mean()) ** 2).sum()
            return 1 - (u / v) if v != 0 else 0.0


# ── Technical Indicators ──────────────────────────────────────────────────────

def compute_sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=window, min_periods=window).mean().round(2)


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI).
    Returns a Series in the range [0, 100].
    """
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()

    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.round(2)


# ── ML Prediction Engine ──────────────────────────────────────────────────────

def build_features(close: np.ndarray, lookback: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a supervised learning dataset from closing prices.

    Features (X): [day_index, rolling_mean_5, rolling_mean_10, rolling_std_5]
    Target  (y): closing price at that day
    """
    n = len(close)
    X, y = [], []

    for i in range(lookback, n):
        window = close[i - lookback: i]
        features = [
            i,                           # linear time index
            np.mean(window[-5:]),        # 5-day mean
            np.mean(window[-10:]),       # 10-day mean
            np.std(window[-5:]),         # 5-day volatility
            window[-1],                  # yesterday's close
            window[-2] if lookback >= 2 else window[-1],  # 2 days ago
        ]
        X.append(features)
        y.append(close[i])

    return np.array(X), np.array(y)


def train_and_predict(df: pd.DataFrame, forecast_days: int = 5) -> dict:
    """
    Train a Linear Regression model on historical closing prices and
    predict the next `forecast_days` closing prices.

    Args:
        df:            DataFrame with at least a 'Close' column.
        forecast_days: Number of future days to predict.

    Returns:
        dict with keys:
            predicted_prices : list[float]  — forecasted closing prices
            r2_score         : float        — model R² on training data
    """
    close    = df["Close"].values.astype(float)
    lookback = 20

    X, y = build_features(close, lookback)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    model = LinearRegression()
    model.fit(X_scaled, y_scaled)

    r2 = model.score(X_scaled, y_scaled)

    # ── Forecast future days ─────────────────────────────────────────────────
    close_extended = list(close)
    predicted: list[float] = []

    for step in range(forecast_days):
        idx    = len(close_extended)
        window = np.array(close_extended[-lookback:])
        feat   = [
            idx,
            np.mean(window[-5:]),
            np.mean(window[-10:]),
            np.std(window[-5:]),
            window[-1],
            window[-2],
        ]
        feat_scaled = scaler_X.transform([feat])
        pred_scaled = model.predict(feat_scaled)
        pred_price  = float(scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0])
        pred_price  = round(pred_price, 2)
        predicted.append(pred_price)
        close_extended.append(pred_price)

    return {"predicted_prices": predicted, "r2_score": round(r2, 4)}


# ── Full Analysis Pipeline ────────────────────────────────────────────────────

def analyse_stock(df: pd.DataFrame, ticker: str) -> dict:
    """
    Run the complete analytical pipeline for one stock.

    Returns a JSON-serialisable dict containing:
        ticker, dates, ohlcv columns, sma50, sma200, rsi,
        predicted_prices, predicted_dates, metrics, raw_r2
    """
    close = df["Close"]

    sma50_series  = compute_sma(close, 50)
    sma200_series = compute_sma(close, 200)
    rsi_series    = compute_rsi(close, 14)

    ml_result = train_and_predict(df, forecast_days=5)

    # ── Latest values ────────────────────────────────────────────────────────
    current_price = round(float(close.iloc[-1]), 2)
    sma50_val     = round(float(sma50_series.iloc[-1]),  2) if not pd.isna(sma50_series.iloc[-1])  else None
    sma200_val    = round(float(sma200_series.iloc[-1]), 2) if not pd.isna(sma200_series.iloc[-1]) else None
    rsi_val       = round(float(rsi_series.iloc[-1]),    2) if not pd.isna(rsi_series.iloc[-1])    else None

    predicted_prices = ml_result["predicted_prices"]

    # ── Predicted dates (next business days) ────────────────────────────────
    last_date = pd.to_datetime(df["Date"].iloc[-1])
    pred_dates = [
        (last_date + pd.offsets.BDay(i + 1)).strftime("%Y-%m-%d")
        for i in range(len(predicted_prices))
    ]

    # ── 5-day change % ───────────────────────────────────────────────────────
    pred_change_pct = round(
        (predicted_prices[-1] - current_price) / current_price * 100, 2
    ) if predicted_prices else 0.0

    # ── NaN-safe series to lists ─────────────────────────────────────────────
    def safe_list(s: pd.Series) -> list:
        return [None if pd.isna(v) else round(float(v), 2) for v in s]

    return {
        "ticker":           ticker,
        "dates":            df["Date"].tolist(),
        "open":             safe_list(df["Open"]),
        "high":             safe_list(df["High"]),
        "low":              safe_list(df["Low"]),
        "close":            safe_list(close),
        "volume":           df["Volume"].tolist(),
        "sma50":            safe_list(sma50_series),
        "sma200":           safe_list(sma200_series),
        "rsi":              safe_list(rsi_series),
        "predicted_prices": predicted_prices,
        "predicted_dates":  pred_dates,
        "metrics": {
            "current_price":    current_price,
            "sma50":            sma50_val,
            "sma200":           sma200_val,
            "rsi":              rsi_val,
            "pred_change_pct":  pred_change_pct,
            "r2_score":         ml_result["r2_score"],
        },
    }
