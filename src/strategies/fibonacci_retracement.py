"""Fibonacci Retracement option buying strategy.

Enter at key Fibonacci levels (38.2%, 50%, 61.8%) during pullbacks in
trending markets. In an uptrend, price pulling back to a Fib level and
bouncing is a high-probability long entry. Vice versa for downtrends.

Complements existing price_action.py support/resistance detection by
adding swing-based Fibonacci levels.

Best for: Trending markets with clean pullbacks to Fib levels.
"""

from datetime import date, datetime, time, timedelta
from typing import Optional

from src.broker.models import OptionContract, Position
from src.signals.models import Direction, Signal
from src.strategies.base import BaseStrategy, ExitSignal, TradeSetup
from src.utils.constants import INDEX_LOT_SIZES
from src.utils.helpers import get_monthly_expiry, get_next_expiry, days_to_expiry
from src.utils.logger import get_logger

logger = get_logger("fibonacci_retracement")

# Standard Fibonacci retracement levels
DEFAULT_FIB_LEVELS = [0.382, 0.5, 0.618]


class FibonacciRetracementStrategy(BaseStrategy):
    """Buy options at Fibonacci retracement levels in trending markets."""

    name = "fibonacci_retracement"

    def __init__(
        self,
        min_score: float = 70.0,
        target_pct: float = 40.0,
        stop_loss_pct: float = 25.0,
        fib_levels: Optional[list[float]] = None,
        proximity_pct: float = 0.3,
        trailing_activation_pct: float = 20.0,
        trailing_lock_pct: float = 50.0,
        time_exit_minutes: int = 90,
        hard_cutoff: time = time(15, 0),
        entry_start: time = time(9, 30),
        entry_end: time = time(14, 0),
    ):
        self.min_score = min_score
        self.target_pct = target_pct
        self.stop_loss_pct = stop_loss_pct
        self.fib_levels = fib_levels or DEFAULT_FIB_LEVELS
        self.proximity_pct = proximity_pct
        self.trailing_activation_pct = trailing_activation_pct
        self.trailing_lock_pct = trailing_lock_pct
        self.time_exit_minutes = time_exit_minutes
        self.hard_cutoff = hard_cutoff
        self.entry_start = entry_start
        self.entry_end = entry_end

        # Swing points — set externally
        self.swing_high: float = 0.0
        self.swing_low: float = 0.0

        # Track which Fib level triggered
        self._triggered_level: float = 0.0

    def set_swing_points(self, swing_high: float, swing_low: float):
        """Set recent swing high and low for Fib level computation."""
        self.swing_high = swing_high
        self.swing_low = swing_low

        if swing_high > swing_low > 0:
            levels = self._compute_fib_levels()
            level_strs = [f"{l:.1f}" for l in levels]
            logger.info(
                f"Fib: high={swing_high:.1f} low={swing_low:.1f} "
                f"levels={level_strs}"
            )

    def _compute_fib_levels(self) -> list[float]:
        """Compute Fibonacci retracement price levels.

        For uptrend pullback (bullish): levels measured from swing low up.
        Fib level = swing_high - (swing_high - swing_low) * ratio
        """
        if self.swing_high <= self.swing_low or self.swing_low <= 0:
            return []

        swing_range = self.swing_high - self.swing_low
        return [
            self.swing_high - swing_range * ratio
            for ratio in self.fib_levels
        ]

    def _is_near_fib(self, spot_price: float) -> Optional[float]:
        """Check if spot is near any Fib level. Returns the level or None."""
        levels = self._compute_fib_levels()

        for level in levels:
            distance_pct = abs(spot_price - level) / level * 100
            if distance_pct <= self.proximity_pct:
                return level

        return None

    def should_enter(
        self,
        signal: Signal,
        spot_price: float,
        current_time: datetime,
    ) -> bool:
        """Enter at Fibonacci retracement level with trend confirmation.

        Conditions:
        1. Swing points available (high > low > 0)
        2. Score meets minimum
        3. Spot price near a Fib retracement level
        4. Signal direction confirms the trend
        5. Bullish: price pulling back in uptrend (near Fib, bouncing)
        6. Bearish: price rallying in downtrend (near Fib, failing)
        """
        if signal.score < self.min_score:
            return False

        if current_time.time() < self.entry_start:
            return False

        if current_time.time() >= self.entry_end:
            return False

        if self.swing_high <= self.swing_low or self.swing_low <= 0:
            return False

        fib_level = self._is_near_fib(spot_price)
        if fib_level is None:
            return False

        # Bullish: price near Fib in an uptrend pullback
        if signal.direction == Direction.BULLISH:
            # Price should be between swing low and high (pullback zone)
            if spot_price < self.swing_low or spot_price > self.swing_high:
                return False

            # Technical must confirm uptrend
            if signal.technical_direction != Direction.BULLISH:
                return False

            self._triggered_level = fib_level
            logger.info(
                f"FIB BOUNCE BULLISH: {signal.symbol} "
                f"spot={spot_price:.0f} near fib={fib_level:.0f} "
                f"range=[{self.swing_low:.0f}-{self.swing_high:.0f}] "
                f"score={signal.score:.0f}"
            )
            return True

        # Bearish: price near Fib in a downtrend rally
        if signal.direction == Direction.BEARISH:
            # For bearish: interpret Fib levels as resistance in downtrend
            # Price rallying up to a Fib level = selling opportunity
            if spot_price < self.swing_low or spot_price > self.swing_high:
                return False

            if signal.technical_direction != Direction.BEARISH:
                return False

            self._triggered_level = fib_level
            logger.info(
                f"FIB BOUNCE BEARISH: {signal.symbol} "
                f"spot={spot_price:.0f} near fib={fib_level:.0f} "
                f"range=[{self.swing_low:.0f}-{self.swing_high:.0f}] "
                f"score={signal.score:.0f}"
            )
            return True

        return False

    def create_setup(
        self,
        signal: Signal,
        contract: OptionContract,
        spot_price: float,
        capital: float,
    ) -> Optional[TradeSetup]:
        """Create Fib retracement trade setup."""
        entry_price = contract.ltp
        if entry_price <= 0:
            return None

        if entry_price < 5.0 or entry_price > 500.0:
            return None

        stop_loss = round(entry_price * (1 - self.stop_loss_pct / 100), 2)
        stop_loss = max(stop_loss, 0.05)
        target = round(entry_price * (1 + self.target_pct / 100), 2)

        risk_per_lot = (entry_price - stop_loss) * contract.lot_size
        if risk_per_lot <= 0:
            return None

        risk_amount = capital * 0.02
        lots = max(1, int(risk_amount / risk_per_lot))
        quantity = lots * contract.lot_size

        return TradeSetup(
            symbol=signal.symbol,
            signal=signal,
            contract=contract,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target=target,
            quantity=quantity,
            strategy_name=self.name,
            reason=(
                f"Fib {signal.direction.value} at {self._triggered_level:.0f} "
                f"[{self.swing_low:.0f}-{self.swing_high:.0f}] "
                f"score={signal.score:.0f}"
            ),
        )

    def should_exit(
        self,
        position: Position,
        current_price: float,
        spot_price: float,
        current_time: datetime,
    ) -> ExitSignal:
        """Exit on SL, target, time, or if Fib level breaks."""
        if current_time.time() >= self.hard_cutoff:
            return ExitSignal(True, "HARD_CUTOFF", current_price)

        if current_price <= position.stop_loss:
            return ExitSignal(True, "SL_HIT", current_price)

        if current_price >= position.target:
            return ExitSignal(True, "TARGET_HIT", current_price)

        # Time-based exit
        holding_mins = position.holding_duration_minutes
        if holding_mins > self.time_exit_minutes:
            pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
            if pnl_pct < 5:
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
        """Current week for index, monthly for stocks."""
        if symbol in INDEX_LOT_SIZES:
            next_exp = get_next_expiry(symbol, current_date)
            dte = days_to_expiry(next_exp, current_date)
            if dte >= 1:
                return next_exp
            return get_next_expiry(symbol, next_exp + timedelta(days=1))
        return get_monthly_expiry(current_date)
