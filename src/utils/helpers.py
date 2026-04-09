"""Helper utilities for market operations, expiry calculations, and more."""

from datetime import date, datetime, time, timedelta
from typing import Optional

from src.utils.constants import (
    DEFAULT_HARD_CUTOFF,
    DEFAULT_TRADING_END,
    DEFAULT_TRADING_START,
    INDEX_EXPIRY_DAYS,
    INDEX_STRIKE_INTERVALS,
    MARKET_CLOSE,
    MARKET_OPEN,
)


def is_market_open(now: Optional[datetime] = None) -> bool:
    """Check if NSE market is currently open."""
    if now is None:
        now = datetime.now()
    if now.weekday() >= 5:  # Saturday/Sunday
        return False
    current_time = now.time()
    return MARKET_OPEN <= current_time <= MARKET_CLOSE


def is_trading_window(
    now: Optional[datetime] = None,
    start: time = DEFAULT_TRADING_START,
    end: time = DEFAULT_TRADING_END,
) -> bool:
    """Check if current time is within the configured trading window."""
    if now is None:
        now = datetime.now()
    if not is_market_open(now):
        return False
    current_time = now.time()
    return start <= current_time <= end


def get_next_expiry(
    symbol: str,
    from_date: Optional[date] = None,
    weekly: bool = True,
) -> date:
    """Calculate the next expiry date for an index/stock.

    For index options with weekly expiry, finds the next occurrence
    of the designated expiry day. For monthly expiry (stocks),
    finds the last Thursday of the current/next month.
    """
    if from_date is None:
        from_date = date.today()

    if weekly and symbol in INDEX_EXPIRY_DAYS:
        expiry_weekday = INDEX_EXPIRY_DAYS[symbol]
        days_ahead = expiry_weekday - from_date.weekday()
        if days_ahead < 0:
            days_ahead += 7
        if days_ahead == 0:
            # If today is expiry day and market hasn't closed, it's today
            now = datetime.now()
            if now.time() <= MARKET_CLOSE:
                return from_date
            days_ahead = 7
        return from_date + timedelta(days=days_ahead)
    else:
        return get_monthly_expiry(from_date)


def get_monthly_expiry(from_date: Optional[date] = None) -> date:
    """Get the last Thursday of the current month (NSE monthly expiry).

    If the last Thursday has passed, returns last Thursday of next month.
    """
    if from_date is None:
        from_date = date.today()

    year, month = from_date.year, from_date.month

    # Find last Thursday of this month
    last_day = _last_day_of_month(year, month)
    last_thursday = last_day
    while last_thursday.weekday() != 3:  # Thursday = 3
        last_thursday -= timedelta(days=1)

    if from_date <= last_thursday:
        return last_thursday

    # Move to next month
    if month == 12:
        year += 1
        month = 1
    else:
        month += 1

    last_day = _last_day_of_month(year, month)
    last_thursday = last_day
    while last_thursday.weekday() != 3:
        last_thursday -= timedelta(days=1)

    return last_thursday


def _last_day_of_month(year: int, month: int) -> date:
    """Get the last day of a given month."""
    if month == 12:
        return date(year + 1, 1, 1) - timedelta(days=1)
    return date(year, month + 1, 1) - timedelta(days=1)


def round_to_strike(price: float, symbol: str) -> int:
    """Round a price to the nearest valid strike price."""
    interval = INDEX_STRIKE_INTERVALS.get(symbol, 50)
    return round(price / interval) * interval


def get_atm_strike(spot_price: float, symbol: str) -> int:
    """Get the at-the-money strike price."""
    return round_to_strike(spot_price, symbol)


def get_otm_strike(
    spot_price: float,
    symbol: str,
    option_type: str,
    steps: int = 2,
) -> int:
    """Get an OTM strike price, N steps away from ATM.

    Args:
        spot_price: Current spot/underlying price
        symbol: Trading symbol
        option_type: 'CE' or 'PE'
        steps: Number of strikes OTM (default: 2)
    """
    interval = INDEX_STRIKE_INTERVALS.get(symbol, 50)
    atm = get_atm_strike(spot_price, symbol)

    if option_type == "CE":
        return atm + (steps * interval)
    else:  # PE
        return atm - (steps * interval)


def days_to_expiry(expiry_date: date, from_date: Optional[date] = None) -> int:
    """Calculate trading days to expiry (approximate)."""
    if from_date is None:
        from_date = date.today()
    calendar_days = (expiry_date - from_date).days
    # Approximate trading days (exclude weekends)
    weeks = calendar_days // 7
    remaining = calendar_days % 7
    trading_days = weeks * 5 + min(remaining, 5)
    return max(trading_days, 0)


def is_expiry_day(
    symbol: str = "NIFTY",
    check_date: Optional[date] = None,
) -> bool:
    """Check if today is an expiry day for the given symbol."""
    if check_date is None:
        check_date = date.today()

    if symbol in INDEX_EXPIRY_DAYS:
        return check_date.weekday() == INDEX_EXPIRY_DAYS[symbol]

    # For stocks, check if it's monthly expiry (last Thursday)
    monthly = get_monthly_expiry(check_date - timedelta(days=1))
    return check_date == monthly


def format_option_symbol(
    symbol: str,
    expiry: date,
    strike: int,
    option_type: str,
) -> str:
    """Format an option symbol for display.

    Example: NIFTY 24APR 22500 CE
    """
    month_str = expiry.strftime("%d%b").upper()
    return f"{symbol} {month_str} {strike} {option_type}"


def calculate_pnl(
    entry_price: float,
    current_price: float,
    quantity: int,
    side: str = "BUY",
) -> float:
    """Calculate P&L for a position."""
    if side == "BUY":
        return (current_price - entry_price) * quantity
    else:
        return (entry_price - current_price) * quantity


def calculate_pnl_pct(entry_price: float, current_price: float, side: str = "BUY") -> float:
    """Calculate P&L percentage."""
    if entry_price == 0:
        return 0.0
    if side == "BUY":
        return ((current_price - entry_price) / entry_price) * 100
    else:
        return ((entry_price - current_price) / entry_price) * 100
