"""OI-based reversal option buying strategy.

Detects massive OI unwinding (short covering / long liquidation) and trades
the reversal with options. Contrarian play combined with S/R confirmation.
"""

from datetime import date, datetime, time
from typing import Optional

from src.broker.models import OptionContract, Position
from src.signals.models import Direction, Signal
from src.strategies.base import BaseStrategy, ExitSignal, TradeSetup
from src.utils.logger import get_logger

logger = get_logger("oi_reversal")


class OIReversalStrategy(BaseStrategy):
    """Buy options on OI-based reversal signals."""

    name = "oi_reversal"

    def __init__(
        self,
        min_score: float = 80.0,
        target_pct: float = 35.0,
        stop_loss_pct: float = 25.0,
        trailing_activation_pct: float = 20.0,
        trailing_lock_pct: float = 50.0,
        hard_cutoff: time = time(15, 0),
    ):
        self.min_score = min_score
        self.target_pct = target_pct
        self.stop_loss_pct = stop_loss_pct
        self.trailing_activation_pct = trailing_activation_pct
        self.trailing_lock_pct = trailing_lock_pct
        self.hard_cutoff = hard_cutoff

    def should_enter(
        self,
        signal: Signal,
        spot_price: float,
        current_time: datetime,
    ) -> bool:
        """Check for OI reversal conditions."""
        if signal.score < self.min_score:
            return False

        if current_time.time() >= self.hard_cutoff:
            return False

        # OI must strongly agree with direction
        if signal.oi_direction != signal.direction:
            return False

        # OI score must be high for reversal confidence
        if signal.oi_score < 70:
            return False

        # Check for unwinding signals in OI details
        has_unwinding = False
        for comp in signal.components:
            if comp.name == "oi_analysis":
                details_lower = comp.details.lower()
                if any(
                    kw in details_lower
                    for kw in ["unwinding", "short covering", "put writing", "call writing"]
                ):
                    has_unwinding = True
                    break

        return has_unwinding

    def create_setup(
        self,
        signal: Signal,
        contract: OptionContract,
        spot_price: float,
        capital: float,
    ) -> Optional[TradeSetup]:
        """Create OI reversal trade setup."""
        entry_price = contract.ltp
        if entry_price <= 0:
            return None

        # Tighter SL for reversal plays (more conviction needed)
        stop_loss = round(entry_price * (1 - self.stop_loss_pct / 100), 2)
        stop_loss = max(stop_loss, 0.05)
        target = round(entry_price * (1 + self.target_pct / 100), 2)

        risk_per_lot = (entry_price - stop_loss) * contract.lot_size
        if risk_per_lot <= 0:
            return None

        # Smaller position for contrarian plays (1.5% risk)
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
            reason=f"OI Reversal {signal.direction.value} OI_score={signal.oi_score:.0f}",
        )

    def should_exit(
        self,
        position: Position,
        current_price: float,
        spot_price: float,
        current_time: datetime,
    ) -> ExitSignal:
        """Check exit conditions."""
        if current_time.time() >= self.hard_cutoff:
            return ExitSignal(True, "HARD_CUTOFF", current_price)

        if current_price <= position.stop_loss:
            return ExitSignal(True, "SL_HIT", current_price)

        if current_price >= position.target:
            return ExitSignal(True, "TARGET_HIT", current_price)

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
