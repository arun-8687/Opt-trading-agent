"""Volume Profile / Point of Control option buying strategy.

Uses Volume Profile levels — POC (Point of Control, highest volume price),
VAH (Value Area High), and VAL (Value Area Low) — as dynamic support and
resistance. Enter on bounce at these levels or breakout through them.

Complements the OI Wall strategy but uses volume distribution instead of
OI concentration. Works for all symbols (not just NIFTY).

Best for: Range-bound days (bounce at POC/VA) or breakout days (break from VA).
"""

from datetime import date, datetime, time, timedelta
from enum import Enum
from typing import Optional

from src.broker.models import OptionContract, Position
from src.signals.models import Direction, Signal
from src.strategies.base import BaseStrategy, ExitSignal, TradeSetup
from src.utils.constants import INDEX_LOT_SIZES
from src.utils.helpers import get_monthly_expiry, get_next_expiry, days_to_expiry
from src.utils.logger import get_logger

logger = get_logger("volume_profile")


class VPAction(Enum):
    BOUNCE = "BOUNCE"  # Bounce off POC/VAH/VAL
    BREAK = "BREAK"    # Break through VAH or VAL


class VolumeProfileStrategy(BaseStrategy):
    """Buy options on Volume Profile level interactions."""

    name = "volume_profile"

    def __init__(
        self,
        min_score: float = 70.0,
        bounce_target_pct: float = 35.0,
        break_target_pct: float = 45.0,
        bounce_sl_pct: float = 25.0,
        break_sl_pct: float = 30.0,
        proximity_pct: float = 0.3,
        trailing_activation_pct: float = 20.0,
        trailing_lock_pct: float = 50.0,
        bounce_time_exit: int = 60,
        break_time_exit: int = 75,
        hard_cutoff: time = time(15, 0),
        entry_start: time = time(9, 45),
        entry_end: time = time(14, 0),
    ):
        self.min_score = min_score
        self.bounce_target_pct = bounce_target_pct
        self.break_target_pct = break_target_pct
        self.bounce_sl_pct = bounce_sl_pct
        self.break_sl_pct = break_sl_pct
        self.proximity_pct = proximity_pct
        self.trailing_activation_pct = trailing_activation_pct
        self.trailing_lock_pct = trailing_lock_pct
        self.bounce_time_exit = bounce_time_exit
        self.break_time_exit = break_time_exit
        self.hard_cutoff = hard_cutoff
        self.entry_start = entry_start
        self.entry_end = entry_end

        # Volume profile data — set externally
        self.poc: float = 0.0
        self.vah: float = 0.0
        self.val: float = 0.0

        # Track detected action
        self._last_action: Optional[VPAction] = None

    def set_volume_profile(self, poc: float, vah: float, val: float):
        """Update volume profile levels (computed from 1-min candle data)."""
        self.poc = poc
        self.vah = vah
        self.val = val

        if poc > 0:
            logger.info(f"VP Levels: POC={poc:.1f} VAH={vah:.1f} VAL={val:.1f}")

    def _is_near(self, price: float, level: float) -> bool:
        """Check if price is within proximity_pct of a level."""
        if level <= 0:
            return False
        distance_pct = abs(price - level) / level * 100
        return distance_pct <= self.proximity_pct

    def _detect_vp_interaction(
        self,
        spot_price: float,
        signal_direction: Direction,
    ) -> tuple[Optional[VPAction], Direction]:
        """Detect if spot is interacting with a VP level."""
        if self.poc <= 0 or self.vah <= 0 or self.val <= 0:
            return None, Direction.NEUTRAL

        # BOUNCE at VAL (support) — buy CE
        if self._is_near(spot_price, self.val) and spot_price >= self.val:
            if signal_direction == Direction.BULLISH:
                return VPAction.BOUNCE, Direction.BULLISH

        # BOUNCE at VAH (resistance) — buy PE
        if self._is_near(spot_price, self.vah) and spot_price <= self.vah:
            if signal_direction == Direction.BEARISH:
                return VPAction.BOUNCE, Direction.BEARISH

        # BOUNCE at POC — direction from signal
        if self._is_near(spot_price, self.poc):
            if signal_direction == Direction.BULLISH:
                return VPAction.BOUNCE, Direction.BULLISH
            if signal_direction == Direction.BEARISH:
                return VPAction.BOUNCE, Direction.BEARISH

        # BREAK above VAH — buy CE (breakout)
        if spot_price > self.vah and not self._is_near(spot_price, self.vah):
            if signal_direction == Direction.BULLISH:
                return VPAction.BREAK, Direction.BULLISH

        # BREAK below VAL — buy PE (breakdown)
        if spot_price < self.val and not self._is_near(spot_price, self.val):
            if signal_direction == Direction.BEARISH:
                return VPAction.BREAK, Direction.BEARISH

        return None, Direction.NEUTRAL

    def should_enter(
        self,
        signal: Signal,
        spot_price: float,
        current_time: datetime,
    ) -> bool:
        """Enter on VP level interaction.

        Conditions:
        1. Volume profile data available
        2. Score meets minimum
        3. Spot near/breaking a VP level
        4. Signal direction confirms the interaction
        """
        if signal.score < self.min_score:
            return False

        if current_time.time() < self.entry_start:
            return False

        if current_time.time() >= self.entry_end:
            return False

        action, direction = self._detect_vp_interaction(spot_price, signal.direction)
        if action is None:
            return False

        self._last_action = action

        logger.info(
            f"VP {action.value}: {signal.symbol} {direction.value} "
            f"POC={self.poc:.0f} VAH={self.vah:.0f} VAL={self.val:.0f} "
            f"spot={spot_price:.0f} score={signal.score:.0f}"
        )
        return True

    def create_setup(
        self,
        signal: Signal,
        contract: OptionContract,
        spot_price: float,
        capital: float,
    ) -> Optional[TradeSetup]:
        """Create setup with bounce/break specific parameters."""
        entry_price = contract.ltp
        if entry_price <= 0:
            return None

        if entry_price < 5.0 or entry_price > 400.0:
            return None

        action = self._last_action or VPAction.BOUNCE

        if action == VPAction.BOUNCE:
            target_pct = self.bounce_target_pct
            sl_pct = self.bounce_sl_pct
        else:
            target_pct = self.break_target_pct
            sl_pct = self.break_sl_pct

        stop_loss = round(entry_price * (1 - sl_pct / 100), 2)
        stop_loss = max(stop_loss, 0.05)
        target = round(entry_price * (1 + target_pct / 100), 2)

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
                f"VP {action.value} {signal.direction.value} "
                f"POC={self.poc:.0f} VAH={self.vah:.0f} VAL={self.val:.0f} "
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
        """Exit with action-specific time limits."""
        if current_time.time() >= self.hard_cutoff:
            return ExitSignal(True, "HARD_CUTOFF", current_price)

        if current_price <= position.stop_loss:
            return ExitSignal(True, "SL_HIT", current_price)

        if current_price >= position.target:
            return ExitSignal(True, "TARGET_HIT", current_price)

        # Time-based exit
        time_limit = self.break_time_exit if self._last_action == VPAction.BREAK else self.bounce_time_exit
        holding_mins = position.holding_duration_minutes
        if holding_mins > time_limit:
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
