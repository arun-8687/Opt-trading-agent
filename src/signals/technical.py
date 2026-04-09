"""Technical indicator-based signal generation.

Calculates RSI, MACD, EMA crossover, SuperTrend, VWAP, ADX, and volume
to produce a directional technical score (0-100).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.signals.models import ComponentScore, Direction
from src.utils.logger import get_logger

logger = get_logger("technical")


@dataclass
class TechnicalConfig:
    """Configuration for technical indicators."""

    ema_fast: int = 9
    ema_slow: int = 21
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    supertrend_period: int = 10
    supertrend_multiplier: float = 3.0
    adx_period: int = 14
    adx_threshold: float = 25.0
    volume_ma_period: int = 20
    volume_surge_ratio: float = 1.5


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calculate_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD line, signal line, and histogram."""
    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_supertrend(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
) -> pd.Series:
    """Calculate SuperTrend indicator.

    Returns a Series where positive values indicate uptrend (bullish)
    and negative values indicate downtrend (bearish).
    """
    hl2 = (df["high"] + df["low"]) / 2

    # ATR calculation
    tr = pd.DataFrame()
    tr["hl"] = df["high"] - df["low"]
    tr["hc"] = abs(df["high"] - df["close"].shift(1))
    tr["lc"] = abs(df["low"] - df["close"].shift(1))
    tr["tr"] = tr[["hl", "hc", "lc"]].max(axis=1)
    atr = tr["tr"].rolling(window=period).mean()

    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)

    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=float)

    for i in range(period, len(df)):
        if i == period:
            supertrend.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = -1
            continue

        if df["close"].iloc[i] > upper_band.iloc[i - 1]:
            direction.iloc[i] = 1
        elif df["close"].iloc[i] < lower_band.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]

        if direction.iloc[i] == 1:
            supertrend.iloc[i] = max(lower_band.iloc[i], supertrend.iloc[i - 1]) \
                if direction.iloc[i - 1] == 1 else lower_band.iloc[i]
        else:
            supertrend.iloc[i] = min(upper_band.iloc[i], supertrend.iloc[i - 1]) \
                if direction.iloc[i - 1] == -1 else upper_band.iloc[i]

    return direction


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average Directional Index."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = pd.DataFrame()
    tr["hl"] = high - low
    tr["hc"] = abs(high - close.shift(1))
    tr["lc"] = abs(low - close.shift(1))
    true_range = tr.max(axis=1)

    atr = true_range.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.rolling(window=period).mean()

    return adx.fillna(0)


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """Calculate Volume Weighted Average Price (intraday)."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cum_vol = df["volume"].cumsum()
    cum_tp_vol = (typical_price * df["volume"]).cumsum()
    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
    return vwap.fillna(typical_price)


def analyze_technical(
    df: pd.DataFrame,
    config: TechnicalConfig = TechnicalConfig(),
) -> ComponentScore:
    """Analyze technical indicators and produce a directional score.

    Args:
        df: DataFrame with columns [timestamp, open, high, low, close, volume]
        config: Technical indicator configuration

    Returns:
        ComponentScore with direction and weighted score (0-100)
    """
    if df.empty or len(df) < config.ema_slow + 5:
        return ComponentScore(
            name="technical",
            direction=Direction.NEUTRAL,
            score=50,
            weight=0.40,
            details="Insufficient data",
        )

    close = df["close"]
    bullish_points = 0
    bearish_points = 0
    total_weight = 0
    details = []

    # 1. EMA Crossover (weight: 15)
    ema_fast = calculate_ema(close, config.ema_fast)
    ema_slow = calculate_ema(close, config.ema_slow)
    ema_fast_curr = ema_fast.iloc[-1]
    ema_slow_curr = ema_slow.iloc[-1]
    ema_fast_prev = ema_fast.iloc[-2]
    ema_slow_prev = ema_slow.iloc[-2]

    if ema_fast_curr > ema_slow_curr:
        bullish_points += 15
        if ema_fast_prev <= ema_slow_prev:
            bullish_points += 5  # Fresh crossover bonus
            details.append("EMA bullish crossover")
        else:
            details.append("EMA bullish")
    elif ema_fast_curr < ema_slow_curr:
        bearish_points += 15
        if ema_fast_prev >= ema_slow_prev:
            bearish_points += 5
            details.append("EMA bearish crossover")
        else:
            details.append("EMA bearish")
    total_weight += 15

    # 2. RSI (weight: 10)
    rsi = calculate_rsi(close, config.rsi_period)
    rsi_curr = rsi.iloc[-1]
    rsi_prev = rsi.iloc[-2]

    if rsi_curr > config.rsi_oversold and rsi_prev <= config.rsi_oversold:
        bullish_points += 10  # Exiting oversold
        details.append(f"RSI exits oversold ({rsi_curr:.0f})")
    elif rsi_curr < config.rsi_overbought and rsi_prev >= config.rsi_overbought:
        bearish_points += 10  # Exiting overbought
        details.append(f"RSI exits overbought ({rsi_curr:.0f})")
    elif rsi_curr > 55:
        bullish_points += 5
        details.append(f"RSI bullish ({rsi_curr:.0f})")
    elif rsi_curr < 45:
        bearish_points += 5
        details.append(f"RSI bearish ({rsi_curr:.0f})")
    total_weight += 10

    # 3. MACD (weight: 15)
    macd_line, signal_line, histogram = calculate_macd(
        close, config.macd_fast, config.macd_slow, config.macd_signal
    )
    macd_curr = macd_line.iloc[-1]
    signal_curr = signal_line.iloc[-1]
    hist_curr = histogram.iloc[-1]
    hist_prev = histogram.iloc[-2]

    if macd_curr > signal_curr:
        bullish_points += 10
        if hist_curr > hist_prev:  # Histogram expanding
            bullish_points += 5
            details.append("MACD bullish + expanding")
        else:
            details.append("MACD bullish")
    elif macd_curr < signal_curr:
        bearish_points += 10
        if hist_curr < hist_prev:
            bearish_points += 5
            details.append("MACD bearish + expanding")
        else:
            details.append("MACD bearish")
    total_weight += 15

    # 4. SuperTrend (weight: 15)
    st_dir = calculate_supertrend(df, config.supertrend_period, config.supertrend_multiplier)
    if not st_dir.empty and pd.notna(st_dir.iloc[-1]):
        if st_dir.iloc[-1] == 1:
            bullish_points += 15
            details.append("SuperTrend bullish")
        elif st_dir.iloc[-1] == -1:
            bearish_points += 15
            details.append("SuperTrend bearish")
    total_weight += 15

    # 5. VWAP (weight: 10)
    vwap = calculate_vwap(df)
    if close.iloc[-1] > vwap.iloc[-1]:
        bullish_points += 10
        if close.iloc[-2] <= vwap.iloc[-2]:
            bullish_points += 3  # Reclaim bonus
            details.append("Price reclaims VWAP")
        else:
            details.append("Price above VWAP")
    elif close.iloc[-1] < vwap.iloc[-1]:
        bearish_points += 10
        if close.iloc[-2] >= vwap.iloc[-2]:
            bearish_points += 3
            details.append("Price breaks VWAP")
        else:
            details.append("Price below VWAP")
    total_weight += 10

    # 6. ADX (weight: 5) — trend strength, not direction
    adx = calculate_adx(df, config.adx_period)
    adx_curr = adx.iloc[-1]
    if adx_curr > config.adx_threshold:
        # Amplify the dominant direction
        if bullish_points > bearish_points:
            bullish_points += 5
        elif bearish_points > bullish_points:
            bearish_points += 5
        details.append(f"ADX trending ({adx_curr:.0f})")
    else:
        details.append(f"ADX weak ({adx_curr:.0f})")
    total_weight += 5

    # 7. Volume confirmation (weight: 10)
    vol_ma = df["volume"].rolling(window=config.volume_ma_period).mean()
    vol_ratio = df["volume"].iloc[-1] / vol_ma.iloc[-1] if vol_ma.iloc[-1] > 0 else 1

    if vol_ratio >= config.volume_surge_ratio:
        if bullish_points > bearish_points:
            bullish_points += 10
        elif bearish_points > bullish_points:
            bearish_points += 10
        details.append(f"Volume surge ({vol_ratio:.1f}x)")
    elif vol_ratio >= 1.0:
        if bullish_points > bearish_points:
            bullish_points += 5
        elif bearish_points > bullish_points:
            bearish_points += 5
        details.append(f"Volume OK ({vol_ratio:.1f}x)")
    else:
        details.append(f"Low volume ({vol_ratio:.1f}x)")
    total_weight += 10

    # Calculate final score
    max_possible = total_weight + 8  # Account for bonus points
    if bullish_points > bearish_points:
        direction = Direction.BULLISH
        score = min(100, (bullish_points / max_possible) * 100)
    elif bearish_points > bullish_points:
        direction = Direction.BEARISH
        score = min(100, (bearish_points / max_possible) * 100)
    else:
        direction = Direction.NEUTRAL
        score = 50

    return ComponentScore(
        name="technical",
        direction=direction,
        score=round(score, 1),
        weight=0.40,
        details=" | ".join(details),
    )
