"""NIFTY Event Day Volatility option buying strategy.

Two-phase strategy for RBI policy, Union Budget, election results,
and other major events. Phase 1: Buy straddle 2-3 days before for
IV expansion. Phase 2: Buy directional after announcement.

NIFTY-specific: RBI affects banking (35%+ of NIFTY weight). Budget
impacts capital gains tax and market sentiment. Historical moves:
Budget 1.5-3%, RBI policy 0.5-1.5%.

Best for: Scheduled macro events (RBI bi-monthly, Budget in February).
"""

from datetime import date, datetime, time, timedelta
from enum import Enum
from typing import Optional

from src.broker.models import OptionContract, Position
from src.signals.models import Direction, Signal
from src.strategies.base import BaseStrategy, ExitSignal, TradeSetup
from src.utils.helpers import get_monthly_expiry, get_next_expiry, days_to_expiry
from src.utils.logger import get_logger

logger = get_logger("nifty_event_day")


class EventPhase(Enum):
    PRE_EVENT = "PRE_EVENT"    # 2-3 days before — buy for IV expansion
    POST_EVENT = "POST_EVENT"  # After announcement — buy direction
    NO_EVENT = "NO_EVENT"      # No event nearby


class NiftyEventDayStrategy(BaseStrategy):
    """Buy NIFTY options around major macro events."""

    name = "nifty_event_day"

    def __init__(
        self,
        min_score: float = 65.0,
        pre_event_target_pct: float = 30.0,
        post_event_target_pct: float = 60.0,
        pre_event_sl_pct: float = 20.0,
        post_event_sl_pct: float = 40.0,
        trailing_activation_pct: float = 25.0,
        trailing_lock_pct: float = 50.0,
        hard_cutoff: time = time(15, 0),
        post_event_entry_start: time = time(10, 0),
    ):
        self.min_score = min_score
        self.pre_event_target_pct = pre_event_target_pct
        self.post_event_target_pct = post_event_target_pct
        self.pre_event_sl_pct = pre_event_sl_pct
        self.post_event_sl_pct = post_event_sl_pct
        self.trailing_activation_pct = trailing_activation_pct
        self.trailing_lock_pct = trailing_lock_pct
        self.hard_cutoff = hard_cutoff
        self.post_event_entry_start = post_event_entry_start

        # Event state — set externally
        self.event_phase: EventPhase = EventPhase.NO_EVENT
        self.event_name: str = ""
        self.event_date: Optional[date] = None
        self.days_to_event: int = -1

    def set_event(self, event_name: str, event_date: date, current_date: date):
        """Configure an upcoming event."""
        self.event_name = event_name
        self.event_date = event_date
        self.days_to_event = (event_date - current_date).days

        if self.days_to_event == 0:
            self.event_phase = EventPhase.POST_EVENT
        elif 1 <= self.days_to_event <= 3:
            self.event_phase = EventPhase.PRE_EVENT
        else:
            self.event_phase = EventPhase.NO_EVENT

        if self.event_phase != EventPhase.NO_EVENT:
            logger.info(
                f"Event: {event_name} on {event_date} "
                f"(D-{self.days_to_event}) phase={self.event_phase.value}"
            )

    def clear_event(self):
        """Clear event state after event passes."""
        self.event_phase = EventPhase.NO_EVENT
        self.event_name = ""
        self.event_date = None
        self.days_to_event = -1

    def should_enter(
        self,
        signal: Signal,
        spot_price: float,
        current_time: datetime,
    ) -> bool:
        """Enter based on event phase.

        Pre-event (D-3 to D-1): Buy if signal is directional
        Post-event (D-day): Buy directional after first 30+ minutes
        """
        if signal.symbol != "NIFTY":
            return False

        if self.event_phase == EventPhase.NO_EVENT:
            return False

        if signal.score < self.min_score:
            return False

        if current_time.time() >= self.hard_cutoff:
            return False

        if signal.direction == Direction.NEUTRAL:
            return False

        if self.event_phase == EventPhase.PRE_EVENT:
            # Pre-event: any directional signal is fine
            # IV expansion will help even moderate signals
            return True

        if self.event_phase == EventPhase.POST_EVENT:
            # Post-event: wait for announcement reaction to settle
            if current_time.time() < self.post_event_entry_start:
                return False

            # Need stronger conviction post-event
            if signal.score < 70:
                return False

            # Technical must confirm the post-event direction
            if signal.technical_direction != signal.direction:
                return False

            return True

        return False

    def create_setup(
        self,
        signal: Signal,
        contract: OptionContract,
        spot_price: float,
        capital: float,
    ) -> Optional[TradeSetup]:
        """Create event trade setup with phase-specific parameters."""
        entry_price = contract.ltp
        if entry_price <= 0:
            return None

        if entry_price < 5.0 or entry_price > 500.0:
            return None

        # Phase-specific parameters
        if self.event_phase == EventPhase.PRE_EVENT:
            target_pct = self.pre_event_target_pct
            sl_pct = self.pre_event_sl_pct
            risk_pct = 0.02  # 2% split between CE+PE legs
        else:
            target_pct = self.post_event_target_pct
            sl_pct = self.post_event_sl_pct
            risk_pct = 0.02

        stop_loss = round(entry_price * (1 - sl_pct / 100), 2)
        stop_loss = max(stop_loss, 0.05)
        target = round(entry_price * (1 + target_pct / 100), 2)

        risk_per_lot = (entry_price - stop_loss) * contract.lot_size
        if risk_per_lot <= 0:
            return None

        risk_amount = capital * risk_pct
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
                f"Event={self.event_name} phase={self.event_phase.value} "
                f"D-{self.days_to_event} {signal.direction.value} "
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
        """Exit — critical: exit ALL positions by end of event day."""
        if current_time.time() >= self.hard_cutoff:
            return ExitSignal(True, "EVENT_DAY_CUTOFF", current_price)

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
        """Pre-event: next week/monthly. Post-event: current week."""
        if self.event_phase == EventPhase.PRE_EVENT:
            # Need time — use next week or monthly
            next_exp = get_next_expiry(symbol, current_date)
            dte = days_to_expiry(next_exp, current_date)
            if dte >= 5:
                return next_exp
            return get_next_expiry(symbol, next_exp + timedelta(days=1))
        else:
            # Post-event: current week for max gamma
            next_exp = get_next_expiry(symbol, current_date)
            dte = days_to_expiry(next_exp, current_date)
            if dte >= 1:
                return next_exp
            return get_next_expiry(symbol, next_exp + timedelta(days=1))
