"""Gap-and-Go option buying strategy.

Trades gap openings (gap up / gap down) that sustain direction after
the first 15-30 minutes. Gaps with volume confirmation and no fill
attempt tend to continue in the gap direction.

Best for: High-conviction gap days with news/event catalyst.
"""

from datetime import date, datetime, time, timedelta
from typing import Optional

from src.broker.models import OptionContract, Position
from src.signals.models import Direction, Signal
from src.strategies.base import BaseStrategy, ExitSignal, TradeSetup
from src.utils.constants import INDEX_LOT_SIZES
from src.utils.helpers import get_monthly_expiry, get_next_expiry, days_to_expiry
from src.utils.logger import get_logger

logger = get_logger("gap_and_go")


class GapAndGoStrategy(BaseStrategy):
    """Buy options in the direction of a sustained gap opening."""

    name = "gap_and_go"

    def __init__(
        self,
        min_score: float = 70.0,
        min_gap_pct: float = 0.5,
        target_pct: float = 45.0,
        stop_loss_pct: float = 30.0,
        trailing_activation_pct: float = 20.0,
        trailing_lock_pct: float = 50.0,
        hard_cutoff: time = time(15, 0),
        entry_start: time = time(9, 35),
        entry_end: time = time(11, 30),
    ):
        self.min_score = min_score
        self.min_gap_pct = min_gap_pct
        self.target_pct = target_pct
        self.stop_loss_pct = stop_loss_pct
        self.trailing_activation_pct = trailing_activation_pct
        self.trailing_lock_pct = trailing_lock_pct
        self.hard_cutoff = hard_cutoff
        self.entry_start = entry_start
        self.entry_end = entry_end

    def should_enter(
        self,
        signal: Signal,
        spot_price: float,
        current_time: datetime,
    ) -> bool:
        """Enter when a gap sustains after opening range.

        Conditions:
        1. Price action shows a gap (gap up sustained / gap down sustained)
        2. Signal direction aligns with the gap direction
        3. Enter after opening range (9:35+) but not too late (before 11:30)
        4. Technical indicators confirm gap direction
        """
        if signal.score < self.min_score:
            return False

        if current_time.time() < self.entry_start:
            return False

        if current_time.time() >= self.entry_end:
            return False

        # Must have a gap signal in price action
        has_gap_sustained = False
        for comp in signal.components:
            if comp.name == "price_action":
                details_lower = comp.details.lower()
                if signal.direction == Direction.BULLISH and "gap up sustained" in details_lower:
                    has_gap_sustained = True
                elif signal.direction == Direction.BEARISH and "gap down sustained" in details_lower:
                    has_gap_sustained = True

        if not has_gap_sustained:
            return False

        # Technical should agree (gap + trend = high conviction)
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
        """Create gap-and-go trade setup."""
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
            reason=f"Gap&Go {signal.direction.value} score={signal.score:.0f}",
        )

    def should_exit(
        self,
        position: Position,
        current_price: float,
        spot_price: float,
        current_time: datetime,
    ) -> ExitSignal:
        """Exit management — if gap fills, exit immediately."""
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

    def select_expiry(self, symbol: str, current_date: date) -> date:
        """Current week for index, monthly for stocks."""
        if symbol in INDEX_LOT_SIZES:
            next_exp = get_next_expiry(symbol, current_date)
            dte = days_to_expiry(next_exp, current_date)
            if dte >= 2:
                return next_exp
            return get_next_expiry(symbol, next_exp + timedelta(days=1))
        return get_monthly_expiry(current_date)
