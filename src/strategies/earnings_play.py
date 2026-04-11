"""Quarterly Results / Earnings option buying strategy.

Two-phase strategy for F&O stocks around quarterly earnings:
Phase 1 (PRE_EARNINGS, D-3 to D-1): Buy options for IV expansion.
Phase 2 (POST_EARNINGS, D-day after 10 AM): Directional play on results gap.

Different from nifty_event_day (macro events like RBI, Budget) — this
targets individual stock results during earnings season (Jan/Apr/Jul/Oct).

Best for: F&O stocks with high options liquidity during results season.
"""

from datetime import date, datetime, time, timedelta
from enum import Enum
from typing import Optional

from src.broker.models import OptionContract, Position
from src.signals.models import Direction, Signal
from src.strategies.base import BaseStrategy, ExitSignal, TradeSetup
from src.utils.constants import FNO_STOCKS, INDEX_LOT_SIZES
from src.utils.helpers import get_monthly_expiry, get_next_expiry, days_to_expiry
from src.utils.logger import get_logger

logger = get_logger("earnings_play")


class EarningsPhase(Enum):
    PRE_EARNINGS = "PRE_EARNINGS"    # D-3 to D-1: IV expansion play
    POST_EARNINGS = "POST_EARNINGS"  # D-day: directional after results
    NO_EARNINGS = "NO_EARNINGS"      # No earnings nearby


class EarningsPlayStrategy(BaseStrategy):
    """Buy options around quarterly earnings announcements."""

    name = "earnings_play"

    def __init__(
        self,
        min_score: float = 68.0,
        pre_target_pct: float = 25.0,
        post_target_pct: float = 60.0,
        pre_sl_pct: float = 15.0,
        post_sl_pct: float = 35.0,
        trailing_activation_pct: float = 20.0,
        trailing_lock_pct: float = 50.0,
        hard_cutoff: time = time(15, 0),
        entry_start: time = time(9, 30),
        entry_end: time = time(14, 0),
        post_entry_start: time = time(10, 0),
        min_gap_pct: float = 2.0,
        pre_time_exit: int = 90,
        post_time_exit: int = 60,
    ):
        self.min_score = min_score
        self.pre_target_pct = pre_target_pct
        self.post_target_pct = post_target_pct
        self.pre_sl_pct = pre_sl_pct
        self.post_sl_pct = post_sl_pct
        self.trailing_activation_pct = trailing_activation_pct
        self.trailing_lock_pct = trailing_lock_pct
        self.hard_cutoff = hard_cutoff
        self.entry_start = entry_start
        self.entry_end = entry_end
        self.post_entry_start = post_entry_start
        self.min_gap_pct = min_gap_pct
        self.pre_time_exit = pre_time_exit
        self.post_time_exit = post_time_exit

        # Earnings state — keyed by symbol, set externally
        self._earnings: dict[str, dict] = {}
        # Track current phase for the symbol being evaluated
        self._current_phase: EarningsPhase = EarningsPhase.NO_EARNINGS
        self._current_symbol: str = ""

    def set_earnings(self, symbol: str, earnings_date: date, current_date: date):
        """Configure upcoming earnings for a stock."""
        days_to = (earnings_date - current_date).days

        if days_to == 0:
            phase = EarningsPhase.POST_EARNINGS
        elif 1 <= days_to <= 3:
            phase = EarningsPhase.PRE_EARNINGS
        else:
            phase = EarningsPhase.NO_EARNINGS

        self._earnings[symbol] = {
            "earnings_date": earnings_date,
            "days_to": days_to,
            "phase": phase,
        }

        if phase != EarningsPhase.NO_EARNINGS:
            logger.info(
                f"Earnings: {symbol} on {earnings_date} "
                f"(D-{days_to}) phase={phase.value}"
            )

    def clear_earnings(self, symbol: str):
        """Clear earnings state after results pass."""
        self._earnings.pop(symbol, None)

    def _get_phase(self, symbol: str) -> EarningsPhase:
        """Get earnings phase for a symbol."""
        info = self._earnings.get(symbol)
        if not info:
            return EarningsPhase.NO_EARNINGS
        return info["phase"]

    def should_enter(
        self,
        signal: Signal,
        spot_price: float,
        current_time: datetime,
    ) -> bool:
        """Enter based on earnings phase.

        Pre-earnings (D-3 to D-1): Buy if signal is directional (IV expansion helps).
        Post-earnings (D-day): Directional play after results gap settles.

        Conditions:
        1. Symbol must be an F&O stock (not index)
        2. Earnings data must be set for the symbol
        3. Signal must be directional (not NEUTRAL)
        """
        # Only F&O stocks, not indices
        if signal.symbol in INDEX_LOT_SIZES:
            return False

        phase = self._get_phase(signal.symbol)
        if phase == EarningsPhase.NO_EARNINGS:
            return False

        if signal.direction == Direction.NEUTRAL:
            return False

        if signal.score < self.min_score:
            return False

        if current_time.time() >= self.hard_cutoff:
            return False

        if phase == EarningsPhase.PRE_EARNINGS:
            # Pre-earnings: any directional signal within entry window
            if current_time.time() < self.entry_start:
                return False
            if current_time.time() >= self.entry_end:
                return False

            self._current_phase = EarningsPhase.PRE_EARNINGS
            self._current_symbol = signal.symbol

            info = self._earnings[signal.symbol]
            logger.info(
                f"EARNINGS PRE: {signal.symbol} D-{info['days_to']} "
                f"{signal.direction.value} score={signal.score:.0f}"
            )
            return True

        if phase == EarningsPhase.POST_EARNINGS:
            # Post-earnings: wait for reaction to settle
            if current_time.time() < self.post_entry_start:
                return False
            if current_time.time() >= self.entry_end:
                return False

            # Need stronger conviction post-earnings
            if signal.score < 75:
                return False

            # Technical must confirm the post-earnings direction
            if signal.technical_direction != signal.direction:
                return False

            # Check for gap in price action components
            has_gap = False
            for comp in signal.components:
                if comp.name == "price_action":
                    details_lower = comp.details.lower()
                    if "gap" in details_lower:
                        has_gap = True

            if not has_gap:
                return False

            self._current_phase = EarningsPhase.POST_EARNINGS
            self._current_symbol = signal.symbol

            logger.info(
                f"EARNINGS POST: {signal.symbol} "
                f"{signal.direction.value} score={signal.score:.0f}"
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
        """Create earnings trade setup with phase-specific parameters."""
        entry_price = contract.ltp
        if entry_price <= 0:
            return None

        if entry_price < 5.0 or entry_price > 500.0:
            return None

        phase = self._current_phase

        if phase == EarningsPhase.PRE_EARNINGS:
            target_pct = self.pre_target_pct
            sl_pct = self.pre_sl_pct
            risk_pct = 0.02
        else:
            target_pct = self.post_target_pct
            sl_pct = self.post_sl_pct
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

        info = self._earnings.get(signal.symbol, {})
        days_to = info.get("days_to", 0)

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
                f"Earnings {phase.value} {signal.direction.value} "
                f"D-{days_to} score={signal.score:.0f}"
            ),
        )

    def should_exit(
        self,
        position: Position,
        current_price: float,
        spot_price: float,
        current_time: datetime,
    ) -> ExitSignal:
        """Exit with phase-specific time limits."""
        if current_time.time() >= self.hard_cutoff:
            return ExitSignal(True, "HARD_CUTOFF", current_price)

        if current_price <= position.stop_loss:
            return ExitSignal(True, "SL_HIT", current_price)

        if current_price >= position.target:
            return ExitSignal(True, "TARGET_HIT", current_price)

        # Time-based exit — use phase-specific limits
        phase = self._get_phase(position.symbol)
        time_limit = self.pre_time_exit if phase == EarningsPhase.PRE_EARNINGS else self.post_time_exit
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
        """Pre-earnings: monthly (time for IV expansion). Post-earnings: current week."""
        phase = self._get_phase(symbol)

        if phase == EarningsPhase.PRE_EARNINGS:
            # Need time — use monthly expiry
            return get_monthly_expiry(current_date)
        else:
            # Post-earnings: current week for max gamma
            if symbol in INDEX_LOT_SIZES:
                next_exp = get_next_expiry(symbol, current_date)
            else:
                next_exp = get_monthly_expiry(current_date)
            dte = days_to_expiry(next_exp, current_date)
            if dte >= 1:
                return next_exp
            return get_monthly_expiry(current_date)
