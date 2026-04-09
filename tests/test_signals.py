"""Tests for signal generation components."""

import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

from src.signals.models import Direction, Signal, SignalStrength, ComponentScore
from src.signals.technical import (
    TechnicalConfig,
    analyze_technical,
    calculate_ema,
    calculate_rsi,
    calculate_macd,
)
from src.signals.signal_engine import SignalEngine


def make_bullish_df(length: int = 50) -> pd.DataFrame:
    """Create a bullish trending DataFrame for testing."""
    dates = [datetime(2024, 1, 1) + timedelta(minutes=5 * i) for i in range(length)]
    base = 22000
    prices = [base + i * 5 + np.random.normal(0, 3) for i in range(length)]

    return pd.DataFrame({
        "timestamp": dates,
        "open": [p - 2 for p in prices],
        "high": [p + 5 for p in prices],
        "low": [p - 5 for p in prices],
        "close": prices,
        "volume": [100000 + i * 1000 for i in range(length)],
    })


def make_bearish_df(length: int = 50) -> pd.DataFrame:
    """Create a bearish trending DataFrame."""
    dates = [datetime(2024, 1, 1) + timedelta(minutes=5 * i) for i in range(length)]
    base = 22000
    prices = [base - i * 5 + np.random.normal(0, 3) for i in range(length)]

    return pd.DataFrame({
        "timestamp": dates,
        "open": [p + 2 for p in prices],
        "high": [p + 5 for p in prices],
        "low": [p - 5 for p in prices],
        "close": prices,
        "volume": [100000 + i * 1000 for i in range(length)],
    })


class TestTechnicalIndicators:
    """Test individual technical indicators."""

    def test_ema_length(self):
        """EMA should return same length as input."""
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        ema = calculate_ema(series, 3)
        assert len(ema) == len(series)

    def test_rsi_range(self):
        """RSI should be between 0 and 100."""
        prices = pd.Series([100 + i * 0.5 for i in range(50)], dtype=float)
        rsi = calculate_rsi(prices, 14)
        assert rsi.dropna().between(0, 100).all()

    def test_rsi_bullish(self):
        """Rising prices should produce RSI >= 50."""
        prices = pd.Series([100 + i for i in range(50)], dtype=float)
        rsi = calculate_rsi(prices, 14)
        assert rsi.iloc[-1] >= 50

    def test_macd_crossover(self):
        """MACD line should cross signal line on trend change."""
        prices = pd.Series([100 + i for i in range(50)], dtype=float)
        macd_line, signal_line, hist = calculate_macd(prices)
        # In a strong uptrend, MACD should be above signal
        assert macd_line.iloc[-1] > signal_line.iloc[-1]


class TestTechnicalAnalysis:
    """Test the complete technical analysis function."""

    def test_bullish_trend_detection(self):
        """Should detect bullish trend in trending data."""
        df = make_bullish_df(60)
        result = analyze_technical(df)
        # Strong bullish trend should be detected
        assert result.direction in (Direction.BULLISH, Direction.NEUTRAL)
        assert result.score > 0

    def test_bearish_trend_detection(self):
        """Should detect bearish trend in falling data."""
        df = make_bearish_df(60)
        result = analyze_technical(df)
        assert result.direction in (Direction.BEARISH, Direction.NEUTRAL)

    def test_insufficient_data(self):
        """Should return neutral for insufficient data."""
        df = make_bullish_df(5)  # Too few candles
        result = analyze_technical(df)
        assert result.direction == Direction.NEUTRAL
        assert result.score == 50

    def test_weight_is_correct(self):
        """Technical weight should be 0.40."""
        df = make_bullish_df(60)
        result = analyze_technical(df)
        assert result.weight == 0.40


class TestSignalEngine:
    """Test the signal combination engine."""

    def test_neutral_without_data(self):
        """Should produce neutral signal without chain data."""
        engine = SignalEngine(min_score=75)
        df = make_bullish_df(10)
        signal = engine.generate_signal(
            symbol="NIFTY",
            intraday_df=df,
            daily_df=df,
            chain_analysis=None,
            spot_price=22000,
        )
        assert isinstance(signal, Signal)
        assert signal.symbol == "NIFTY"

    def test_signal_has_components(self):
        """Signal should have component scores."""
        engine = SignalEngine()
        df = make_bullish_df(60)
        signal = engine.generate_signal(
            symbol="NIFTY",
            intraday_df=df,
            daily_df=df,
            chain_analysis=None,
            spot_price=22000,
        )
        assert signal.technical_score >= 0
        assert signal.oi_score >= 0
        assert signal.iv_score >= 0
        assert signal.price_action_score >= 0

    def test_cooldown_enforcement(self):
        """Should enforce cooldown between signals for same symbol."""
        engine = SignalEngine(min_score=0, cooldown_minutes=15)
        # Manually set last signal time
        engine._last_signal_time["NIFTY"] = datetime.now()

        df = make_bullish_df(60)
        signal = engine.generate_signal(
            symbol="NIFTY",
            intraday_df=df,
            daily_df=df,
            chain_analysis=None,
            spot_price=22000,
        )
        assert signal.direction == Direction.NEUTRAL
        assert "Cooldown" in signal.details

    def test_cooldown_reset(self):
        """Should allow resetting cooldown."""
        engine = SignalEngine()
        engine._last_signal_time["NIFTY"] = datetime.now()
        engine.reset_cooldown("NIFTY")
        assert "NIFTY" not in engine._last_signal_time
