"""Straddle breakout option buying strategy.

Buys an ATM straddle (both CE and PE) and rides the winning leg
when a significant directional move occurs. The losing leg is
exited quickly while the winning leg trails to maximize profit.

Alternatively, waits for the straddle range to establish then
buys only the breakout direction.

Best for: Pre-event uncertainty, range-bound markets about to break.
"""

from datetime import date, datetime, time, timedelta
from typing import Optional

from src.broker.models import OptionContract, OptionType, Position
from src.signals.models import Direction, Signal
from src.strategies.base import BaseStrategy, ExitSignal, TradeSetup
from src.utils.constants import INDEX_LOT_SIZES, INDEX_STRIKE_INTERVALS
from src.utils.helpers import (
    days_to_expiry,
    get_atm_strike,
    get_monthly_expiry,
    get_next_expiry,
)
from src.utils.logger import get_logger

logger = get_logger("straddle_breakout")


class StraddleBreakoutStrategy(BaseStrategy):
    """Buy options on straddle range breakout.

    Instead of buying both legs, we wait for the ATM straddle's
    combined premium to indicate a range, then buy the directional
    leg once price breaks out of the range.
    """

    name = "straddle_breakout"

    def __init__(
        self,
        min_score: float = 70.0,
        target_pct: float = 50.0,
        stop_loss_pct: float = 35.0,
        trailing_activation_pct: float = 25.0,
        trailing_lock_pct: float = 55.0,
        time_exit_minutes: int = 90,
        hard_cutoff: time = time(15, 0),
        entry_start: time = time(9, 45),
        entry_end: time = time(13, 30),
    ):
        self.min_score = min_score
        self.target_pct = target_pct
        self.stop_loss_pct = stop_loss_pct
        self.trailing_activation_pct = trailing_activation_pct
        self.trailing_lock_pct = trailing_lock_pct
        self.time_exit_minutes = time_exit_minutes
        self.hard_cutoff = hard_cutoff
        self.entry_start = entry_start
        self.entry_end = entry_end

    def should_enter(
        self,
        signal: Signal,
        spot_price: float,
        current_time: datetime,
    ) -> bool:
        """Enter when price breaks out of opening range with signal confirmation.

        Straddle breakout conditions:
        1. Price action shows ORB breakout or PDH/PDL break
        2. At least 2 signal categories agree
        3. After initial consolidation (9:45+)
        4. Score meets minimum threshold
        """
        if signal.score < self.min_score:
            return False

        if current_time.time() < self.entry_start:
            return False

        if current_time.time() >= self.entry_end:
            return False

        if signal.agreement_count < 2:
            return False

        # Need price action showing a range breakout
        has_range_break = False
        for comp in signal.components:
            if comp.name == "price_action":
                details_lower = comp.details.lower()
                if any(
                    kw in details_lower
                    for kw in [
                        "orb breakout", "orb breakdown",
                        "above pdh", "below pdl",
                        "above r1", "below s1",
                    ]
                ):
                    has_range_break = True

        if not has_range_break:
            return False

        # IV should not be extremely high (we're buying)
        if signal.iv_score < 30:
            # IV is very unfavorable for buying
            return False

        return True

    def create_setup(
        self,
        signal: Signal,
        contract: OptionContract,
        spot_price: float,
        capital: float,
    ) -> Optional[TradeSetup]:
        """Create straddle breakout setup.

        Higher target since breakouts can run significantly.
        Wider SL to avoid getting stopped on retest of breakout level.
        """
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
                f"Straddle breakout {signal.direction.value} "
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
        """Exit management for straddle breakout."""
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
        """Current week expiry — breakouts benefit from high gamma."""
        if symbol in INDEX_LOT_SIZES:
            next_exp = get_next_expiry(symbol, current_date)
            dte = days_to_expiry(next_exp, current_date)
            if dte >= 1:
                return next_exp
            return get_next_expiry(symbol, next_exp + timedelta(days=1))
        return get_monthly_expiry(current_date)
