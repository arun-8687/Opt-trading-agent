"""Implied Volatility-based signal analysis.

Analyzes IV Rank, IV Percentile, and IV skew to determine
whether options are cheap/expensive and detect directional bias.
"""

import pandas as pd

from src.data.options_chain import OptionsChainAnalysis
from src.signals.models import ComponentScore, Direction
from src.utils.logger import get_logger

logger = get_logger("iv_analysis")


def calculate_iv_rank(current_iv: float, iv_history: pd.Series) -> float:
    """Calculate IV Rank (where current IV sits in the historical range).

    IV Rank = (Current IV - Min IV) / (Max IV - Min IV) * 100
    """
    if iv_history.empty or len(iv_history) < 5:
        return 50.0

    min_iv = iv_history.min()
    max_iv = iv_history.max()
    iv_range = max_iv - min_iv

    if iv_range <= 0:
        return 50.0

    return ((current_iv - min_iv) / iv_range) * 100


def calculate_iv_percentile(current_iv: float, iv_history: pd.Series) -> float:
    """Calculate IV Percentile (% of days IV was below current level).

    IV Percentile = (Days IV was below current) / Total days * 100
    """
    if iv_history.empty or len(iv_history) < 5:
        return 50.0

    below_count = (iv_history < current_iv).sum()
    return (below_count / len(iv_history)) * 100


def analyze_iv(
    chain_analysis: OptionsChainAnalysis,
    iv_history: pd.Series = pd.Series(dtype=float),
) -> ComponentScore:
    """Analyze IV metrics for option buying suitability.

    For option buying:
    - Low IV (IV Rank < 30): Options are cheap → FAVORABLE for buying
    - High IV (IV Rank > 70): Options are expensive → UNFAVORABLE for buying
    - IV Skew: Can indicate directional bias

    Args:
        chain_analysis: Current options chain analysis with IV data
        iv_history: Historical ATM IV values (last 252 trading days ideally)

    Returns:
        ComponentScore with IV-based assessment
    """
    current_iv = chain_analysis.atm_iv
    iv_skew = chain_analysis.iv_skew

    if current_iv <= 0:
        return ComponentScore(
            name="iv_analysis",
            direction=Direction.NEUTRAL,
            score=50,
            weight=0.15,
            details="No IV data",
        )

    bullish_points = 0
    bearish_points = 0
    details = []

    # 1. IV Rank / IV Level Analysis (weight: 10)
    # For option buying, we WANT low IV (cheap options)
    if not iv_history.empty and len(iv_history) >= 20:
        iv_rank = calculate_iv_rank(current_iv, iv_history)
        iv_pctile = calculate_iv_percentile(current_iv, iv_history)

        if iv_rank < 30:
            # Low IV = options are cheap = good for buying
            # This doesn't indicate direction, but boosts confidence
            bullish_points += 5
            bearish_points += 5
            details.append(f"IV Rank LOW ({iv_rank:.0f}) - cheap options")
        elif iv_rank < 50:
            bullish_points += 3
            bearish_points += 3
            details.append(f"IV Rank moderate ({iv_rank:.0f})")
        elif iv_rank < 70:
            bullish_points += 1
            bearish_points += 1
            details.append(f"IV Rank elevated ({iv_rank:.0f})")
        else:
            # High IV = expensive options = bad for buying
            # Reduce scores to penalize
            details.append(f"IV Rank HIGH ({iv_rank:.0f}) - expensive options")
    else:
        # No history - use absolute IV levels
        if current_iv < 0.15:
            bullish_points += 5
            bearish_points += 5
            details.append(f"IV low ({current_iv:.1%})")
        elif current_iv < 0.25:
            bullish_points += 3
            bearish_points += 3
            details.append(f"IV moderate ({current_iv:.1%})")
        else:
            details.append(f"IV high ({current_iv:.1%})")

    # 2. IV Skew Analysis (weight: 5)
    # Positive skew (CE IV > PE IV): Market expects upside → bearish contrarian
    # Negative skew (PE IV > CE IV): Market expects downside → bullish contrarian
    # Actually for buying: follow the skew (not contrarian)
    if abs(iv_skew) > 0.01:
        if iv_skew > 0.02:
            # CE IV higher = demand for calls = bullish sentiment
            bullish_points += 5
            details.append(f"IV skew bullish ({iv_skew:.3f})")
        elif iv_skew < -0.02:
            # PE IV higher = demand for puts = bearish sentiment
            bearish_points += 5
            details.append(f"IV skew bearish ({iv_skew:.3f})")
        else:
            details.append(f"IV skew neutral ({iv_skew:.3f})")
    else:
        details.append("IV skew flat")

    # Calculate score
    total_possible = 15
    if bullish_points > bearish_points:
        direction = Direction.BULLISH
        score = min(100, (bullish_points / total_possible) * 100)
    elif bearish_points > bullish_points:
        direction = Direction.BEARISH
        score = min(100, (bearish_points / total_possible) * 100)
    else:
        direction = Direction.NEUTRAL
        score = max(50, min(100, (bullish_points / total_possible) * 100))

    return ComponentScore(
        name="iv_analysis",
        direction=direction,
        score=round(score, 1),
        weight=0.15,
        details=" | ".join(details),
    )
