"""Stock filtering criteria for the scanner."""

from dataclasses import dataclass

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("filters")


@dataclass
class ScannerFilterConfig:
    """Configuration for stock scanner filters."""

    min_option_oi: int = 5000
    min_volume_ratio: float = 1.5
    min_adx: float = 20.0
    volume_ma_period: int = 20
    min_price: float = 50.0
    max_price: float = 50000.0
    min_avg_volume: int = 100000  # Minimum average daily volume in shares


def filter_by_liquidity(
    df: pd.DataFrame,
    min_avg_volume: int = 100000,
) -> pd.DataFrame:
    """Filter stocks by minimum average trading volume.

    Args:
        df: DataFrame with 'symbol' and 'avg_volume' columns
        min_avg_volume: Minimum average daily volume threshold
    """
    if "avg_volume" not in df.columns:
        return df
    return df[df["avg_volume"] >= min_avg_volume].copy()


def filter_by_volume_surge(
    df: pd.DataFrame,
    min_ratio: float = 1.5,
) -> pd.DataFrame:
    """Filter stocks showing volume surge vs their average.

    Args:
        df: DataFrame with 'volume' and 'avg_volume' columns
        min_ratio: Minimum current_volume / avg_volume ratio
    """
    if "volume" not in df.columns or "avg_volume" not in df.columns:
        return df
    df = df[df["avg_volume"] > 0].copy()
    df["volume_ratio"] = df["volume"] / df["avg_volume"]
    return df[df["volume_ratio"] >= min_ratio].copy()


def filter_by_trend_strength(
    df: pd.DataFrame,
    min_adx: float = 20.0,
) -> pd.DataFrame:
    """Filter stocks with sufficient trend strength (ADX).

    Args:
        df: DataFrame with 'adx' column
        min_adx: Minimum ADX value to consider stock trending
    """
    if "adx" not in df.columns:
        return df
    return df[df["adx"] >= min_adx].copy()


def filter_by_price_range(
    df: pd.DataFrame,
    min_price: float = 50.0,
    max_price: float = 50000.0,
) -> pd.DataFrame:
    """Filter stocks within a price range.

    Args:
        df: DataFrame with 'ltp' column
        min_price: Minimum stock price
        max_price: Maximum stock price
    """
    if "ltp" not in df.columns:
        return df
    return df[(df["ltp"] >= min_price) & (df["ltp"] <= max_price)].copy()


def filter_by_change(
    df: pd.DataFrame,
    min_abs_change_pct: float = 0.5,
) -> pd.DataFrame:
    """Filter stocks showing meaningful price movement.

    Args:
        df: DataFrame with 'change_pct' column
        min_abs_change_pct: Minimum absolute % change
    """
    if "change_pct" not in df.columns:
        return df
    return df[abs(df["change_pct"]) >= min_abs_change_pct].copy()


def rank_stocks(df: pd.DataFrame) -> pd.DataFrame:
    """Rank filtered stocks by combined score for trading priority.

    Higher rank = better candidate. Considers volume surge,
    trend strength, and momentum.
    """
    if df.empty:
        return df

    df = df.copy()

    # Normalize each factor to 0-1
    for col in ["volume_ratio", "adx", "abs_change_pct"]:
        if col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()
            col_range = col_max - col_min
            if col_range > 0:
                df[f"{col}_norm"] = (df[col] - col_min) / col_range
            else:
                df[f"{col}_norm"] = 0.5

    # Calculate composite score
    score = pd.Series(0.0, index=df.index)
    if "volume_ratio_norm" in df.columns:
        score += df["volume_ratio_norm"] * 0.35
    if "adx_norm" in df.columns:
        score += df["adx_norm"] * 0.35
    if "abs_change_pct_norm" in df.columns:
        score += df["abs_change_pct_norm"] * 0.30

    df["scan_score"] = score
    return df.sort_values("scan_score", ascending=False)
