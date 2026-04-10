"""NIFTY OI Wall Bounce/Break option buying strategy.

The strike with highest put OI acts as NIFTY support (put writers
defend it). The strike with highest call OI acts as resistance.
Buy on bounce at these levels, or buy the break when they fail.

NIFTY-specific: NIFTY's total OI (50+ lakh crore notional) makes
OI walls extremely meaningful. When a wall breaks, delta-hedging
cascades create violent 75-150 point moves.

Best for: Tue-Wed when OI is stable, range-bound weeks with
clear OI walls.
"""

from datetime import date, datetime, time, timedelta
from enum import Enum
from typing import Optional

from src.broker.models import OptionContract, Position
from src.signals.models import Direction, Signal
from src.strategies.base import BaseStrategy, ExitSignal, TradeSetup
from src.utils.helpers import get_next_expiry, days_to_expiry, is_expiry_day
from src.utils.logger import get_logger

logger = get_logger("nifty_oi_wall")


class OIWallAction(Enum):
    BOUNCE = "BOUNCE"  # Bounce off OI wall (support/resistance holds)
    BREAK = "BREAK"    # Break through OI wall (cascade move)


class NiftyOIWallStrategy(BaseStrategy):
    """Buy NIFTY options on OI wall interactions."""

    name = "nifty_oi_wall"

    def __init__(
        self,
        min_score: float = 68.0,
        bounce_target_pct: float = 35.0,
        break_target_pct: float = 50.0,
        bounce_sl_pct: float = 25.0,
        break_sl_pct: float = 30.0,
        trailing_activation_pct: float = 20.0,
        trailing_lock_pct: float = 55.0,
        bounce_time_exit: int = 60,
        break_time_exit: int = 90,
        hard_cutoff: time = time(15, 0),
        entry_start: time = time(9, 45),
        entry_end: time = time(14, 0),
        proximity_points: float = 50.0,
    ):
        self.min_score = min_score
        self.bounce_target_pct = bounce_target_pct
        self.break_target_pct = break_target_pct
        self.bounce_sl_pct = bounce_sl_pct
        self.break_sl_pct = break_sl_pct
        self.trailing_activation_pct = trailing_activation_pct
        self.trailing_lock_pct = trailing_lock_pct
        self.bounce_time_exit = bounce_time_exit
        self.break_time_exit = break_time_exit
        self.hard_cutoff = hard_cutoff
        self.entry_start = entry_start
        self.entry_end = entry_end
        self.proximity_points = proximity_points

        # OI wall data — set externally
        self.max_put_oi_strike: int = 0
        self.max_call_oi_strike: int = 0
        self.put_oi_at_wall: int = 0
        self.call_oi_at_wall: int = 0

        # Track detected action for current signal
        self._last_action: Optional[OIWallAction] = None

    def set_oi_walls(
        self,
        max_put_oi_strike: int,
        max_call_oi_strike: int,
        put_oi: int = 0,
        call_oi: int = 0,
    ):
        """Update OI wall levels."""
        self.max_put_oi_strike = max_put_oi_strike
        self.max_call_oi_strike = max_call_oi_strike
        self.put_oi_at_wall = put_oi
        self.call_oi_at_wall = call_oi

        if max_put_oi_strike > 0 and max_call_oi_strike > 0:
            logger.info(
                f"OI Walls: Put={max_put_oi_strike} (OI={put_oi:,}) "
                f"Call={max_call_oi_strike} (OI={call_oi:,})"
            )

    def _detect_wall_interaction(
        self,
        spot_price: float,
        signal_direction: Direction,
    ) -> tuple[Optional[OIWallAction], Direction]:
        """Detect if spot is near/breaking an OI wall."""
        if self.max_put_oi_strike <= 0 or self.max_call_oi_strike <= 0:
            return None, Direction.NEUTRAL

        dist_to_put_wall = spot_price - self.max_put_oi_strike
        dist_to_call_wall = self.max_call_oi_strike - spot_price

        # BOUNCE at put wall (support) — buy CE
        if 0 < dist_to_put_wall <= self.proximity_points:
            if signal_direction == Direction.BULLISH:
                return OIWallAction.BOUNCE, Direction.BULLISH

        # BOUNCE at call wall (resistance) — buy PE
        if 0 < dist_to_call_wall <= self.proximity_points:
            if signal_direction == Direction.BEARISH:
                return OIWallAction.BOUNCE, Direction.BEARISH

        # BREAK below put wall — buy PE (cascading fall)
        if spot_price < self.max_put_oi_strike:
            if signal_direction == Direction.BEARISH:
                return OIWallAction.BREAK, Direction.BEARISH

        # BREAK above call wall — buy CE (short covering rally)
        if spot_price > self.max_call_oi_strike:
            if signal_direction == Direction.BULLISH:
                return OIWallAction.BREAK, Direction.BULLISH

        return None, Direction.NEUTRAL

    def should_enter(
        self,
        signal: Signal,
        spot_price: float,
        current_time: datetime,
    ) -> bool:
        """Enter on OI wall bounce or break.

        Conditions:
        1. NIFTY only
        2. OI wall data available
        3. Spot near a wall (bounce) or breaking through (break)
        4. Signal direction confirms the interaction
        5. Not on expiry day (OI shifts too rapidly)
        """
        if signal.symbol != "NIFTY":
            return False

        if signal.score < self.min_score:
            return False

        if current_time.time() < self.entry_start:
            return False

        if current_time.time() >= self.entry_end:
            return False

        # Skip expiry day — OI levels are unreliable
        if is_expiry_day("NIFTY", current_time):
            return False

        action, direction = self._detect_wall_interaction(spot_price, signal.direction)
        if action is None:
            return False

        self._last_action = action

        wall_strike = (
            self.max_put_oi_strike
            if direction == Direction.BULLISH
            else self.max_call_oi_strike
        )
        logger.info(
            f"OI WALL {action.value}: NIFTY {direction.value} "
            f"at wall={wall_strike} spot={spot_price:.0f} "
            f"score={signal.score:.0f}"
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

        action = self._last_action or OIWallAction.BOUNCE

        if action == OIWallAction.BOUNCE:
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

        wall_info = (
            f"put_wall={self.max_put_oi_strike} call_wall={self.max_call_oi_strike}"
        )

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
                f"OI Wall {action.value} {signal.direction.value} "
                f"{wall_info} score={signal.score:.0f}"
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

        # Use longer time exit for breaks (cascade takes time)
        time_limit = self.break_time_exit if self._last_action == OIWallAction.BREAK else self.bounce_time_exit
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
        """Bounce: current week. Break: also current week (max gamma)."""
        next_exp = get_next_expiry(symbol, current_date)
        dte = days_to_expiry(next_exp, current_date)
        if dte >= 1:
            return next_exp
        return get_next_expiry(symbol, next_exp + timedelta(days=1))
