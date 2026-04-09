"""Breakout/breakdown option buying strategy.

Entry: Price breaks previous day high/low or opening range with volume surge.
Confirmation: OI supports the move, SuperTrend aligned.
Target: 50% of premium. Stop Loss: 30%.
"""

from datetime import date, datetime, time
from typing import Optional

from src.broker.models import OptionContract, Position
from src.signals.models import Direction, Signal
from src.strategies.base import BaseStrategy, ExitSignal, TradeSetup
from src.utils.logger import get_logger

logger = get_logger("breakout_buy")


class BreakoutBuyStrategy(BaseStrategy):
    """Buy options on breakout/breakdown of key levels."""

    name = "breakout_buy"

    def __init__(
        self,
        min_score: float = 78.0,
        target_pct: float = 50.0,
        stop_loss_pct: float = 30.0,
        volume_surge_ratio: float = 2.0,
        trailing_activation_pct: float = 25.0,
        trailing_lock_pct: float = 50.0,
        hard_cutoff: time = time(15, 0),
    ):
        self.min_score = min_score
        self.target_pct = target_pct
        self.stop_loss_pct = stop_loss_pct
        self.volume_surge_ratio = volume_surge_ratio
        self.trailing_activation_pct = trailing_activation_pct
        self.trailing_lock_pct = trailing_lock_pct
        self.hard_cutoff = hard_cutoff

    def should_enter(
        self,
        signal: Signal,
        spot_price: float,
        current_time: datetime,
    ) -> bool:
        """Check if signal represents a valid breakout."""
        if signal.score < self.min_score:
            return False

        if signal.agreement_count < 3:
            return False

        if current_time.time() >= self.hard_cutoff:
            return False

        # Breakout requires strong technical + price action alignment
        if signal.technical_direction != signal.direction:
            return False
        if signal.price_action_direction != signal.direction:
            return False

        # Check for "breakout" keywords in price action details
        has_breakout = False
        for comp in signal.components:
            if comp.name == "price_action":
                details_lower = comp.details.lower()
                if any(
                    kw in details_lower
                    for kw in ["breakout", "breakdown", "above pdh", "below pdl", "orb"]
                ):
                    has_breakout = True
                    break

        if not has_breakout:
            return False

        # Check for volume surge in technical details
        for comp in signal.components:
            if comp.name == "technical" and "volume surge" in comp.details.lower():
                return True

        # Accept even without volume surge if score is very high
        return signal.score >= 85

    def create_setup(
        self,
        signal: Signal,
        contract: OptionContract,
        spot_price: float,
        capital: float,
    ) -> Optional[TradeSetup]:
        """Create breakout trade setup."""
        entry_price = contract.ltp
        if entry_price <= 0:
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
            reason=f"Breakout {signal.direction.value} score={signal.score:.0f}",
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
