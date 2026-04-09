"""Expiry day directional option buying strategy.

Trades the accelerated theta decay and directional momentum on expiry day.
ATM/slight-OTM options can move 50-200% on expiry day.
Very tight SL and quick targets.
"""

from datetime import date, datetime, time
from typing import Optional

from src.broker.models import OptionContract, Position
from src.signals.models import Signal
from src.strategies.base import BaseStrategy, ExitSignal, TradeSetup
from src.utils.helpers import is_expiry_day
from src.utils.logger import get_logger

logger = get_logger("expiry_day")


class ExpiryDayStrategy(BaseStrategy):
    """Specialized strategy for weekly expiry day trading."""

    name = "expiry_day"

    def __init__(
        self,
        min_score: float = 72.0,
        target_pct: float = 75.0,
        stop_loss_pct: float = 40.0,
        opening_range_minutes: int = 15,
        trailing_activation_pct: float = 30.0,
        trailing_lock_pct: float = 60.0,
        entry_start: time = time(9, 35),
        entry_end: time = time(13, 0),
        hard_cutoff: time = time(14, 30),
    ):
        self.min_score = min_score
        self.target_pct = target_pct
        self.stop_loss_pct = stop_loss_pct
        self.opening_range_minutes = opening_range_minutes
        self.trailing_activation_pct = trailing_activation_pct
        self.trailing_lock_pct = trailing_lock_pct
        self.entry_start = entry_start
        self.entry_end = entry_end
        self.hard_cutoff = hard_cutoff

    def should_enter(
        self,
        signal: Signal,
        spot_price: float,
        current_time: datetime,
    ) -> bool:
        """Check if conditions are right for expiry day trade."""
        # Must be an expiry day
        if not is_expiry_day(signal.symbol, current_time.date()):
            return False

        if signal.score < self.min_score:
            return False

        # Trade only after opening range settles
        if current_time.time() < self.entry_start:
            return False

        # Don't enter too late on expiry
        if current_time.time() >= self.entry_end:
            return False

        # Need at least technical alignment
        if signal.technical_direction != signal.direction:
            return False

        return True

    def create_setup(
        self,
        signal: Signal,
        contract: OptionContract,
        spot_price: float,
        capital: float,
    ) -> Optional[TradeSetup]:
        """Create expiry day trade setup.

        Uses wider stop but also higher target (options move big on expiry).
        Smaller position size to manage the higher volatility.
        """
        entry_price = contract.ltp
        if entry_price <= 0:
            return None

        # Expiry day: wider SL (options spike before dying)
        stop_loss = round(entry_price * (1 - self.stop_loss_pct / 100), 2)
        stop_loss = max(stop_loss, 0.05)

        # Higher target (options can double on expiry day)
        target = round(entry_price * (1 + self.target_pct / 100), 2)

        risk_per_lot = (entry_price - stop_loss) * contract.lot_size
        if risk_per_lot <= 0:
            return None

        # Smaller size for expiry day (1.5% risk)
        risk_amount = capital * 0.015
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
            reason=f"Expiry day {signal.direction.value} score={signal.score:.0f}",
        )

    def should_exit(
        self,
        position: Position,
        current_price: float,
        spot_price: float,
        current_time: datetime,
    ) -> ExitSignal:
        """Aggressive exit management for expiry day."""
        # Very strict hard cutoff on expiry (options can go to 0)
        if current_time.time() >= self.hard_cutoff:
            return ExitSignal(True, "EXPIRY_CUTOFF", current_price)

        if current_price <= position.stop_loss:
            return ExitSignal(True, "SL_HIT", current_price)

        if current_price >= position.target:
            return ExitSignal(True, "TARGET_HIT", current_price)

        # Aggressive trailing on expiry day
        if current_price > position.high_price:
            position.high_price = current_price

        profit_pct = ((position.high_price - position.entry_price) / position.entry_price) * 100
        if profit_pct >= self.trailing_activation_pct:
            trail_sl = position.entry_price + (
                (position.high_price - position.entry_price) * (self.trailing_lock_pct / 100)
            )
            if trail_sl > position.stop_loss:
                position.stop_loss = round(trail_sl, 2)

        # If option is near zero, exit immediately
        if current_price < 1.0 and position.entry_price > 5.0:
            return ExitSignal(True, "NEAR_ZERO", current_price)

        return ExitSignal(False)

    def select_expiry(self, symbol: str, current_date: date) -> date:
        """On expiry day, trade the expiring contract."""
        return current_date
