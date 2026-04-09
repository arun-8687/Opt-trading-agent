"""RSI divergence option buying strategy.

Detects bullish and bearish RSI divergences — when price makes new
highs/lows but RSI doesn't confirm. These are powerful reversal signals
that indicate weakening momentum before a direction change.

Best for: Range-bound to mildly trending markets, catching reversals.
"""

from datetime import date, datetime, time, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from src.broker.models import OptionContract, Position
from src.signals.models import Direction, Signal
from src.signals.technical import calculate_rsi
from src.strategies.base import BaseStrategy, ExitSignal, TradeSetup
from src.utils.constants import INDEX_LOT_SIZES
from src.utils.helpers import get_monthly_expiry, get_next_expiry, days_to_expiry
from src.utils.logger import get_logger

logger = get_logger("rsi_divergence")


def detect_rsi_divergence(
    df: pd.DataFrame,
    rsi_period: int = 14,
    lookback: int = 20,
) -> tuple[Direction, str]:
    """Detect bullish or bearish RSI divergence.

    Bullish divergence: Price makes lower low, RSI makes higher low
    Bearish divergence: Price makes higher high, RSI makes lower high

    Args:
        df: DataFrame with close prices
        rsi_period: RSI calculation period
        lookback: Number of candles to look back for divergence

    Returns:
        (direction, description) tuple
    """
    if len(df) < rsi_period + lookback:
        return Direction.NEUTRAL, "Insufficient data"

    close = df["close"]
    rsi = calculate_rsi(close, rsi_period)

    recent = lookback
    price_slice = close.iloc[-recent:]
    rsi_slice = rsi.iloc[-recent:]

    # Find swing lows (for bullish divergence)
    price_lows = []
    rsi_lows = []
    for i in range(2, len(price_slice) - 2):
        if (
            price_slice.iloc[i] < price_slice.iloc[i - 1]
            and price_slice.iloc[i] < price_slice.iloc[i - 2]
            and price_slice.iloc[i] < price_slice.iloc[i + 1]
            and price_slice.iloc[i] < price_slice.iloc[i + 2]
        ):
            price_lows.append((i, price_slice.iloc[i]))
            rsi_lows.append((i, rsi_slice.iloc[i]))

    # Bullish divergence: price lower low + RSI higher low
    if len(price_lows) >= 2:
        last_price_low = price_lows[-1][1]
        prev_price_low = price_lows[-2][1]
        last_rsi_low = rsi_lows[-1][1]
        prev_rsi_low = rsi_lows[-2][1]

        if last_price_low < prev_price_low and last_rsi_low > prev_rsi_low:
            return Direction.BULLISH, (
                f"Bullish RSI divergence: price LL ({last_price_low:.0f} < {prev_price_low:.0f}) "
                f"RSI HL ({last_rsi_low:.0f} > {prev_rsi_low:.0f})"
            )

    # Find swing highs (for bearish divergence)
    price_highs = []
    rsi_highs = []
    for i in range(2, len(price_slice) - 2):
        if (
            price_slice.iloc[i] > price_slice.iloc[i - 1]
            and price_slice.iloc[i] > price_slice.iloc[i - 2]
            and price_slice.iloc[i] > price_slice.iloc[i + 1]
            and price_slice.iloc[i] > price_slice.iloc[i + 2]
        ):
            price_highs.append((i, price_slice.iloc[i]))
            rsi_highs.append((i, rsi_slice.iloc[i]))

    # Bearish divergence: price higher high + RSI lower high
    if len(price_highs) >= 2:
        last_price_high = price_highs[-1][1]
        prev_price_high = price_highs[-2][1]
        last_rsi_high = rsi_highs[-1][1]
        prev_rsi_high = rsi_highs[-2][1]

        if last_price_high > prev_price_high and last_rsi_high < prev_rsi_high:
            return Direction.BEARISH, (
                f"Bearish RSI divergence: price HH ({last_price_high:.0f} > {prev_price_high:.0f}) "
                f"RSI LH ({last_rsi_high:.0f} < {prev_rsi_high:.0f})"
            )

    return Direction.NEUTRAL, "No divergence"


class RSIDivergenceStrategy(BaseStrategy):
    """Buy options on confirmed RSI divergence reversals."""

    name = "rsi_divergence"

    def __init__(
        self,
        min_score: float = 68.0,
        rsi_period: int = 14,
        lookback: int = 20,
        target_pct: float = 40.0,
        stop_loss_pct: float = 25.0,
        trailing_activation_pct: float = 20.0,
        trailing_lock_pct: float = 50.0,
        time_exit_minutes: int = 90,
        hard_cutoff: time = time(15, 0),
    ):
        self.min_score = min_score
        self.rsi_period = rsi_period
        self.lookback = lookback
        self.target_pct = target_pct
        self.stop_loss_pct = stop_loss_pct
        self.trailing_activation_pct = trailing_activation_pct
        self.trailing_lock_pct = trailing_lock_pct
        self.time_exit_minutes = time_exit_minutes
        self.hard_cutoff = hard_cutoff

        # Stored divergence info from last check
        self._last_divergence: dict[str, tuple[Direction, str]] = {}

    def check_divergence(self, symbol: str, df: pd.DataFrame) -> tuple[Direction, str]:
        """Check and cache divergence for a symbol."""
        div_dir, div_desc = detect_rsi_divergence(df, self.rsi_period, self.lookback)
        self._last_divergence[symbol] = (div_dir, div_desc)
        return div_dir, div_desc

    def should_enter(
        self,
        signal: Signal,
        spot_price: float,
        current_time: datetime,
    ) -> bool:
        """Enter when RSI divergence aligns with overall signal direction.

        Divergence alone can be early — we combine with the signal engine
        to filter for higher probability entries.
        """
        if signal.score < self.min_score:
            return False

        if current_time.time() >= self.hard_cutoff:
            return False

        # Check if we have a divergence for this symbol
        div_data = self._last_divergence.get(signal.symbol)
        if not div_data:
            return False

        div_direction, div_desc = div_data

        if div_direction == Direction.NEUTRAL:
            return False

        # Divergence direction must match overall signal
        if div_direction != signal.direction:
            return False

        # OI should support (or at least not oppose) the reversal
        if signal.oi_direction != Direction.NEUTRAL and signal.oi_direction != signal.direction:
            return False

        logger.info(f"RSI Divergence detected for {signal.symbol}: {div_desc}")
        return True

    def create_setup(
        self,
        signal: Signal,
        contract: OptionContract,
        spot_price: float,
        capital: float,
    ) -> Optional[TradeSetup]:
        """Create divergence trade setup.

        Slightly tighter SL since divergence gives a clear invalidation level.
        """
        entry_price = contract.ltp
        if entry_price <= 0:
            return None

        if entry_price < 5.0 or entry_price > 400.0:
            return None

        stop_loss = round(entry_price * (1 - self.stop_loss_pct / 100), 2)
        stop_loss = max(stop_loss, 0.05)
        target = round(entry_price * (1 + self.target_pct / 100), 2)

        risk_per_lot = (entry_price - stop_loss) * contract.lot_size
        if risk_per_lot <= 0:
            return None

        # Smaller size for reversal (1.5% risk — contrarian play)
        risk_amount = capital * 0.015
        lots = max(1, int(risk_amount / risk_per_lot))
        quantity = lots * contract.lot_size

        div_desc = ""
        if signal.symbol in self._last_divergence:
            div_desc = self._last_divergence[signal.symbol][1]

        return TradeSetup(
            symbol=signal.symbol,
            signal=signal,
            contract=contract,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target=target,
            quantity=quantity,
            strategy_name=self.name,
            reason=f"RSI Divergence: {div_desc}",
        )

    def should_exit(
        self,
        position: Position,
        current_price: float,
        spot_price: float,
        current_time: datetime,
    ) -> ExitSignal:
        """Exit management for divergence trades."""
        if current_time.time() >= self.hard_cutoff:
            return ExitSignal(True, "HARD_CUTOFF", current_price)

        if current_price <= position.stop_loss:
            return ExitSignal(True, "SL_HIT", current_price)

        if current_price >= position.target:
            return ExitSignal(True, "TARGET_HIT", current_price)

        # Longer time allowance for divergence (reversals take time)
        holding_mins = position.holding_duration_minutes
        if holding_mins > self.time_exit_minutes:
            pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
            if pnl_pct < -5:
                return ExitSignal(
                    True,
                    f"TIME_EXIT ({holding_mins:.0f}min, P&L={pnl_pct:.1f}%)",
                    current_price,
                )

        # Trailing SL
        if current_price > position.high_price:
            position.high_price = current_price

        profit_pct = ((position.high_price - position.entry_price) / position.entry_price) * 100
        if profit_pct >= self.trailing_activation_pct:
            trail_sl = position.entry_price + (
                (position.high_price - position.entry_price) * (self.trailing_lock_pct / 100)
            )
            if trail_sl > position.stop_loss:
                position.stop_loss = round(trail_sl, 2)

        return ExitSignal(False)

    def select_expiry(self, symbol: str, current_date: date) -> date:
        """Prefer next week — divergence reversals need time to play out."""
        if symbol in INDEX_LOT_SIZES:
            next_exp = get_next_expiry(symbol, current_date)
            dte = days_to_expiry(next_exp, current_date)
            if dte >= 3:
                return next_exp
            return get_next_expiry(symbol, next_exp + timedelta(days=1))
        return get_monthly_expiry(current_date)
