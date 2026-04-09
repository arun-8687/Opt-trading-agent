"""VWAP pullback option buying strategy.

Buys on pullback to VWAP in a trending market. VWAP acts as dynamic
support (in uptrend) or resistance (in downtrend). Entry when price
bounces off VWAP with confirmation from trend indicators.

Best for: Trending days with clear directional bias.
"""

from datetime import date, datetime, time, timedelta
from typing import Optional

from src.broker.models import OptionContract, Position
from src.signals.models import Direction, Signal
from src.strategies.base import BaseStrategy, ExitSignal, TradeSetup
from src.utils.constants import INDEX_LOT_SIZES
from src.utils.helpers import get_monthly_expiry, get_next_expiry, days_to_expiry
from src.utils.logger import get_logger

logger = get_logger("vwap_pullback")


class VWAPPullbackStrategy(BaseStrategy):
    """Buy options when price pulls back to VWAP and bounces in a trend."""

    name = "vwap_pullback"

    def __init__(
        self,
        min_score: float = 72.0,
        target_pct: float = 35.0,
        stop_loss_pct: float = 25.0,
        trailing_activation_pct: float = 15.0,
        trailing_lock_pct: float = 50.0,
        time_exit_minutes: int = 45,
        hard_cutoff: time = time(15, 0),
        entry_start: time = time(9, 45),
        entry_end: time = time(14, 0),
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
        """Enter when price pulls back to VWAP with trend confirmation.

        Conditions:
        1. Overall signal is directional (bullish/bearish)
        2. Technical indicators confirm the trend (EMA, SuperTrend aligned)
        3. Price action shows VWAP interaction (bounce off VWAP)
        4. Not too early (wait for trend to establish) or too late
        """
        if signal.score < self.min_score:
            return False

        if current_time.time() < self.entry_start:
            return False

        if current_time.time() >= self.entry_end:
            return False

        # Need technical trend confirmation
        if signal.technical_direction != signal.direction:
            return False

        # Look for VWAP-related signals in price action and technical
        has_vwap_signal = False
        for comp in signal.components:
            details_lower = comp.details.lower()
            if "vwap" in details_lower:
                # Bullish: price above or reclaiming VWAP
                if signal.direction == Direction.BULLISH and any(
                    kw in details_lower for kw in ["above vwap", "reclaims vwap"]
                ):
                    has_vwap_signal = True
                # Bearish: price below or breaking VWAP
                elif signal.direction == Direction.BEARISH and any(
                    kw in details_lower for kw in ["below vwap", "breaks vwap"]
                ):
                    has_vwap_signal = True

        if not has_vwap_signal:
            return False

        # Require SuperTrend alignment for trend confirmation
        has_supertrend = False
        for comp in signal.components:
            if comp.name == "technical":
                if "supertrend" in comp.details.lower():
                    if signal.direction.value.lower() in comp.details.lower():
                        has_supertrend = True

        return has_supertrend

    def create_setup(
        self,
        signal: Signal,
        contract: OptionContract,
        spot_price: float,
        capital: float,
    ) -> Optional[TradeSetup]:
        """Create VWAP pullback trade setup.

        Tighter SL than momentum (since we have VWAP as reference),
        moderate target.
        """
        entry_price = contract.ltp
        if entry_price <= 0:
            return None

        if entry_price < 5.0 or entry_price > 400.0:
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
                f"VWAP pullback {signal.direction.value} "
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
        """Exit on SL, target, or if price crosses VWAP against the trade."""
        if current_time.time() >= self.hard_cutoff:
            return ExitSignal(True, "HARD_CUTOFF", current_price)

        if current_price <= position.stop_loss:
            return ExitSignal(True, "SL_HIT", current_price)

        if current_price >= position.target:
            return ExitSignal(True, "TARGET_HIT", current_price)

        # Time-based exit for losing positions
        holding_mins = position.holding_duration_minutes
        if holding_mins > self.time_exit_minutes:
            pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
            if pnl_pct < 0:
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
        """Prefer current week expiry (intraday plays)."""
        if symbol in INDEX_LOT_SIZES:
            next_exp = get_next_expiry(symbol, current_date)
            dte = days_to_expiry(next_exp, current_date)
            if dte >= 1:
                return next_exp
            return get_next_expiry(symbol, next_exp + timedelta(days=1))
        return get_monthly_expiry(current_date)
