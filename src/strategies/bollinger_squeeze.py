"""Bollinger Band Squeeze option buying strategy.

Detects low-volatility compression via Bollinger Band squeeze (bandwidth
below threshold), then enters on breakout when bands expand. Tight squeezes
often precede explosive moves.

Different from straddle_breakout — this uses BB bandwidth specifically to
detect volatility compression rather than ATM straddle premium.

Best for: Consolidation phases where a big move is imminent.
"""

from datetime import date, datetime, time, timedelta
from typing import Optional

from src.broker.models import OptionContract, Position
from src.signals.models import Direction, Signal
from src.strategies.base import BaseStrategy, ExitSignal, TradeSetup
from src.utils.constants import INDEX_LOT_SIZES
from src.utils.helpers import get_monthly_expiry, get_next_expiry, days_to_expiry
from src.utils.logger import get_logger

logger = get_logger("bollinger_squeeze")


class BollingerSqueezeStrategy(BaseStrategy):
    """Buy options on Bollinger Band squeeze breakout."""

    name = "bollinger_squeeze"

    def __init__(
        self,
        min_score: float = 72.0,
        target_pct: float = 45.0,
        stop_loss_pct: float = 30.0,
        max_bandwidth: float = 0.04,
        min_expansion_ratio: float = 1.5,
        trailing_activation_pct: float = 20.0,
        trailing_lock_pct: float = 50.0,
        time_exit_minutes: int = 75,
        hard_cutoff: time = time(15, 0),
        entry_start: time = time(9, 30),
        entry_end: time = time(14, 0),
    ):
        self.min_score = min_score
        self.target_pct = target_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_bandwidth = max_bandwidth
        self.min_expansion_ratio = min_expansion_ratio
        self.trailing_activation_pct = trailing_activation_pct
        self.trailing_lock_pct = trailing_lock_pct
        self.time_exit_minutes = time_exit_minutes
        self.hard_cutoff = hard_cutoff
        self.entry_start = entry_start
        self.entry_end = entry_end

        # BB data — set externally
        self.upper: float = 0.0
        self.lower: float = 0.0
        self.middle: float = 0.0
        self.bandwidth: float = 0.0
        self.prev_bandwidth: float = 0.0

    def set_bb_data(
        self,
        upper: float,
        lower: float,
        middle: float,
        bandwidth: float,
        prev_bandwidth: float,
    ):
        """Update Bollinger Band data.

        bandwidth = (upper - lower) / middle
        """
        self.upper = upper
        self.lower = lower
        self.middle = middle
        self.bandwidth = bandwidth
        self.prev_bandwidth = prev_bandwidth

        if bandwidth > 0:
            logger.debug(
                f"BB: upper={upper:.1f} lower={lower:.1f} mid={middle:.1f} "
                f"bw={bandwidth:.4f} prev_bw={prev_bandwidth:.4f}"
            )

    def should_enter(
        self,
        signal: Signal,
        spot_price: float,
        current_time: datetime,
    ) -> bool:
        """Enter on squeeze breakout.

        Conditions:
        1. BB data available
        2. Previous bandwidth was in squeeze (below threshold)
        3. Current bandwidth expanding (ratio above minimum)
        4. Price breaking above upper (bullish) or below lower (bearish)
        5. Signal direction confirms the breakout
        """
        if signal.score < self.min_score:
            return False

        if current_time.time() < self.entry_start:
            return False

        if current_time.time() >= self.entry_end:
            return False

        # BB data must be available
        if self.middle <= 0 or self.prev_bandwidth <= 0:
            return False

        # Squeeze must have been present
        if self.prev_bandwidth > self.max_bandwidth:
            return False

        # Expansion must be happening
        if self.prev_bandwidth > 0:
            expansion = self.bandwidth / self.prev_bandwidth
        else:
            return False

        if expansion < self.min_expansion_ratio:
            return False

        # Breakout direction must match signal
        if signal.direction == Direction.BULLISH and spot_price > self.upper:
            logger.info(
                f"BB SQUEEZE BREAKOUT UP: {signal.symbol} "
                f"spot={spot_price:.0f} > upper={self.upper:.0f} "
                f"bw_expansion={expansion:.1f}x score={signal.score:.0f}"
            )
            return True

        if signal.direction == Direction.BEARISH and spot_price < self.lower:
            logger.info(
                f"BB SQUEEZE BREAKOUT DOWN: {signal.symbol} "
                f"spot={spot_price:.0f} < lower={self.lower:.0f} "
                f"bw_expansion={expansion:.1f}x score={signal.score:.0f}"
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
        """Create squeeze breakout trade setup."""
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

        expansion = self.bandwidth / self.prev_bandwidth if self.prev_bandwidth > 0 else 0

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
                f"BB Squeeze breakout {signal.direction.value} "
                f"expansion={expansion:.1f}x score={signal.score:.0f}"
            ),
        )

    def should_exit(
        self,
        position: Position,
        current_price: float,
        spot_price: float,
        current_time: datetime,
    ) -> ExitSignal:
        """Exit on SL, target, or re-entry into bands."""
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
