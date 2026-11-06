"""
insights.py
-----------
Natural language insight generator that synthesises technical indicators and
ML predictions into concise, human-readable financial commentary.
"""

from __future__ import annotations


def generate_insight(
    ticker: str,
    current_price: float,
    sma50: float | None,
    sma200: float | None,
    predicted_prices: list[float],
    rsi: float | None = None,
) -> dict:
    """
    Build a structured insight object from technical + ML signals.

    Returns a dict with keys:
        sentiment   : "Bullish" | "Bearish" | "Neutral"
        summary     : one-sentence headline
        details     : list of bullet-point strings
        opportunity : actionable suggestion string
    """
    details: list[str] = []
    signals: list[str] = []

    # ── SMA signals ──────────────────────────────────────────────────────────
    if sma50 is not None and sma200 is not None:
        if sma50 > sma200:
            signals.append("bullish")
            details.append(
                f"📈 Golden Cross detected: 50-day SMA (${sma50:.2f}) is above "
                f"200-day SMA (${sma200:.2f}), signalling bullish momentum."
            )
        else:
            signals.append("bearish")
            details.append(
                f"📉 Death Cross detected: 50-day SMA (${sma50:.2f}) is below "
                f"200-day SMA (${sma200:.2f}), signalling bearish pressure."
            )

    # ── Price vs SMA50 ───────────────────────────────────────────────────────
    if sma50 is not None:
        pct_vs_sma50 = (current_price - sma50) / sma50 * 100
        if pct_vs_sma50 > 2:
            signals.append("bullish")
            details.append(
                f"💹 Price is trading {pct_vs_sma50:.1f}% above its 50-day SMA — "
                "short-term uptrend intact."
            )
        elif pct_vs_sma50 < -2:
            signals.append("bearish")
            details.append(
                f"⚠️  Price is {abs(pct_vs_sma50):.1f}% below its 50-day SMA — "
                "short-term downtrend in play."
            )

    # ── ML prediction signal ─────────────────────────────────────────────────
    if predicted_prices:
        last_pred   = predicted_prices[-1]
        pred_change = (last_pred - current_price) / current_price * 100
        direction   = "gain" if pred_change >= 0 else "loss"
        signals.append("bullish" if pred_change >= 0 else "bearish")
        details.append(
            f"🤖 ML model (Linear Regression) forecasts a {abs(pred_change):.2f}% "
            f"{direction} over the next 5 trading days "
            f"(target: ${last_pred:.2f})."
        )

    # ── RSI signal ───────────────────────────────────────────────────────────
    if rsi is not None:
        if rsi >= 70:
            signals.append("bearish")
            details.append(
                f"🔴 RSI at {rsi:.1f} — stock is in overbought territory; "
                "watch for potential pullback."
            )
        elif rsi <= 30:
            signals.append("bullish")
            details.append(
                f"🟢 RSI at {rsi:.1f} — stock is oversold; "
                "reversal opportunity possible."
            )
        else:
            details.append(f"⚪ RSI at {rsi:.1f} — neutral momentum zone.")

    # ── Aggregate sentiment ──────────────────────────────────────────────────
    bullish_count = signals.count("bullish")
    bearish_count = signals.count("bearish")

    if bullish_count > bearish_count:
        sentiment = "Bullish"
    elif bearish_count > bullish_count:
        sentiment = "Bearish"
    else:
        sentiment = "Neutral"

    # ── Summary headline ─────────────────────────────────────────────────────
    pred_pct_str = ""
    if predicted_prices:
        pred_change = (predicted_prices[-1] - current_price) / current_price * 100
        pred_pct_str = f" Predicted {abs(pred_change):.1f}% {'gain' if pred_change >= 0 else 'loss'} over next 5 days."

    if sentiment == "Bullish":
        summary = (
            f"{ticker} shows bullish momentum based on SMA crossover analysis and ML forecasting.{pred_pct_str}"
        )
        opportunity = (
            "Consider monitoring for a confirmed breakout above recent resistance levels before entering a long position."
        )
    elif sentiment == "Bearish":
        summary = (
            f"{ticker} exhibits bearish signals. Exercise caution with new long positions.{pred_pct_str}"
        )
        opportunity = (
            "Risk management is key — consider setting stop-loss orders or waiting for stabilisation signals."
        )
    else:
        summary = (
            f"{ticker} is in a consolidation phase with mixed signals.{pred_pct_str}"
        )
        opportunity = (
            "The stock is range-bound. Watch for a breakout in either direction before committing capital."
        )

    return {
        "sentiment":   sentiment,
        "summary":     summary,
        "details":     details,
        "opportunity": opportunity,
    }
