"""Price action signal generation.

Analyzes support/resistance levels, breakouts, previous day high/low,
opening range, and gap analysis for directional signals.
"""

import pandas as pd

from src.signals.models import ComponentScore, Direction
from src.utils.logger import get_logger

logger = get_logger("price_action")


def calculate_pivot_points(
    high: float,
    low: float,
    close: float,
) -> dict[str, float]:
    """Calculate standard pivot points from previous day's data.

    Returns dict with keys: pp, r1, r2, r3, s1, s2, s3
    """
    pp = (high + low + close) / 3
    r1 = 2 * pp - low
    s1 = 2 * pp - high
    r2 = pp + (high - low)
    s2 = pp - (high - low)
    r3 = high + 2 * (pp - low)
    s3 = low - 2 * (high - pp)

    return {
        "pp": round(pp, 2),
        "r1": round(r1, 2),
        "r2": round(r2, 2),
        "r3": round(r3, 2),
        "s1": round(s1, 2),
        "s2": round(s2, 2),
        "s3": round(s3, 2),
    }


def detect_opening_range(
    intraday_df: pd.DataFrame,
    range_minutes: int = 15,
) -> tuple[float, float]:
    """Detect opening range (high/low of first N minutes).

    Returns (opening_range_high, opening_range_low)
    """
    if intraday_df.empty:
        return 0.0, 0.0

    first_ts = intraday_df["timestamp"].iloc[0]
    cutoff = first_ts + pd.Timedelta(minutes=range_minutes)
    opening_candles = intraday_df[intraday_df["timestamp"] <= cutoff]

    if opening_candles.empty:
        return 0.0, 0.0

    return float(opening_candles["high"].max()), float(opening_candles["low"].min())


def analyze_price_action(
    intraday_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    current_price: float,
) -> ComponentScore:
    """Analyze price action patterns for directional bias.

    Args:
        intraday_df: Today's intraday candles (1min or 5min)
        daily_df: Recent daily candles (last 20+ days)
        current_price: Current LTP of the underlying

    Returns:
        ComponentScore with price action-based direction and score
    """
    if intraday_df.empty or daily_df.empty or current_price <= 0:
        return ComponentScore(
            name="price_action",
            direction=Direction.NEUTRAL,
            score=50,
            weight=0.15,
            details="Insufficient data",
        )

    bullish_points = 0
    bearish_points = 0
    details = []

    # 1. Previous Day High/Low (weight: 5)
    prev_day = daily_df.iloc[-1] if len(daily_df) >= 1 else None
    if prev_day is not None:
        pdh = float(prev_day["high"])
        pdl = float(prev_day["low"])
        pdc = float(prev_day["close"])

        if current_price > pdh:
            bullish_points += 5
            details.append(f"Above PDH ({pdh:.0f})")
        elif current_price < pdl:
            bearish_points += 5
            details.append(f"Below PDL ({pdl:.0f})")
        elif current_price > pdc:
            bullish_points += 2
            details.append("Above prev close")
        elif current_price < pdc:
            bearish_points += 2
            details.append("Below prev close")

    # 2. Pivot Points (weight: 5)
    if len(daily_df) >= 1:
        prev = daily_df.iloc[-1]
        pivots = calculate_pivot_points(
            float(prev["high"]), float(prev["low"]), float(prev["close"])
        )

        if current_price > pivots["r1"]:
            bullish_points += 5
            details.append(f"Above R1 ({pivots['r1']:.0f})")
        elif current_price > pivots["pp"]:
            bullish_points += 3
            details.append(f"Above pivot ({pivots['pp']:.0f})")
        elif current_price < pivots["s1"]:
            bearish_points += 5
            details.append(f"Below S1 ({pivots['s1']:.0f})")
        elif current_price < pivots["pp"]:
            bearish_points += 3
            details.append(f"Below pivot ({pivots['pp']:.0f})")

    # 3. Opening Range Breakout (weight: 5)
    orb_high, orb_low = detect_opening_range(intraday_df, 15)
    if orb_high > 0 and orb_low > 0:
        if current_price > orb_high:
            bullish_points += 5
            details.append(f"ORB breakout ({orb_high:.0f})")
        elif current_price < orb_low:
            bearish_points += 5
            details.append(f"ORB breakdown ({orb_low:.0f})")
        else:
            details.append("Within opening range")

    # 4. Gap Analysis (weight: 3)
    if len(daily_df) >= 1 and not intraday_df.empty:
        prev_close = float(daily_df.iloc[-1]["close"])
        today_open = float(intraday_df.iloc[0]["open"])
        gap_pct = ((today_open - prev_close) / prev_close) * 100

        if gap_pct > 0.5:
            # Gap up
            if current_price > today_open:
                bullish_points += 3  # Gap up sustained
                details.append(f"Gap up sustained (+{gap_pct:.1f}%)")
            else:
                bearish_points += 2  # Gap fill attempt
                details.append(f"Gap up filling (+{gap_pct:.1f}%)")
        elif gap_pct < -0.5:
            # Gap down
            if current_price < today_open:
                bearish_points += 3
                details.append(f"Gap down sustained ({gap_pct:.1f}%)")
            else:
                bullish_points += 2
                details.append(f"Gap down filling ({gap_pct:.1f}%)")

    # 5. Intraday trend (weight: 2)
    if len(intraday_df) >= 5:
        recent = intraday_df.tail(5)
        higher_lows = all(
            recent["low"].iloc[i] >= recent["low"].iloc[i - 1]
            for i in range(1, len(recent))
        )
        lower_highs = all(
            recent["high"].iloc[i] <= recent["high"].iloc[i - 1]
            for i in range(1, len(recent))
        )

        if higher_lows:
            bullish_points += 2
            details.append("Higher lows forming")
        elif lower_highs:
            bearish_points += 2
            details.append("Lower highs forming")

    # Calculate score
    total_possible = 20
    if bullish_points > bearish_points:
        direction = Direction.BULLISH
        score = min(100, (bullish_points / total_possible) * 100)
    elif bearish_points > bullish_points:
        direction = Direction.BEARISH
        score = min(100, (bearish_points / total_possible) * 100)
    else:
        direction = Direction.NEUTRAL
        score = 50

    return ComponentScore(
        name="price_action",
        direction=direction,
        score=round(score, 1),
        weight=0.15,
        details=" | ".join(details),
    )
