"""Scalping / quick trade option buying strategy.

Very short-term trades targeting 15-25% profit on premium with
tight 15% stop loss. Enters on strong momentum bursts with high
volume confirmation. Holds for 5-20 minutes typically.

Best for: High-volatility intraday sessions, event days.
"""

from datetime import date, datetime, time, timedelta
from typing import Optional

from src.broker.models import OptionContract, Position
from src.signals.models import Direction, Signal, SignalStrength
from src.strategies.base import BaseStrategy, ExitSignal, TradeSetup
from src.utils.constants import INDEX_LOT_SIZES
from src.utils.helpers import get_next_expiry, days_to_expiry
from src.utils.logger import get_logger

logger = get_logger("scalping")


class ScalpingStrategy(BaseStrategy):
    """Quick in-and-out option trades on strong momentum bursts."""

    name = "scalping"

    def __init__(
        self,
        min_score: float = 80.0,
        target_pct: float = 20.0,
        stop_loss_pct: float = 15.0,
        trailing_activation_pct: float = 10.0,
        trailing_lock_pct: float = 60.0,
        max_hold_minutes: int = 20,
        hard_cutoff: time = time(15, 0),
        entry_start: time = time(9, 20),
        entry_end: time = time(14, 30),
        min_premium: float = 10.0,
        max_premium: float = 300.0,
    ):
        self.min_score = min_score
        self.target_pct = target_pct
        self.stop_loss_pct = stop_loss_pct
        self.trailing_activation_pct = trailing_activation_pct
        self.trailing_lock_pct = trailing_lock_pct
        self.max_hold_minutes = max_hold_minutes
        self.hard_cutoff = hard_cutoff
        self.entry_start = entry_start
        self.entry_end = entry_end
        self.min_premium = min_premium
        self.max_premium = max_premium

    def should_enter(
        self,
        signal: Signal,
        spot_price: float,
        current_time: datetime,
    ) -> bool:
        """Enter on strong momentum with volume surge.

        Scalping requires:
        1. Very high signal score (>= 80) for conviction
        2. Technical direction clear and strong
        3. Volume surge (mentioned in technical details)
        4. Quick entry window
        """
        if signal.score < self.min_score:
            return False

        if signal.strength != SignalStrength.STRONG:
            return False

        if current_time.time() < self.entry_start:
            return False

        if current_time.time() >= self.entry_end:
            return False

        # Must have strong technical alignment
        if signal.technical_direction != signal.direction:
            return False

        if signal.technical_score < 70:
            return False

        # Volume confirmation is essential for scalping
        has_volume = False
        for comp in signal.components:
            if comp.name == "technical" and "volume" in comp.details.lower():
                if "surge" in comp.details.lower() or "ok" in comp.details.lower():
                    has_volume = True

        return has_volume

    def create_setup(
        self,
        signal: Signal,
        contract: OptionContract,
        spot_price: float,
        capital: float,
    ) -> Optional[TradeSetup]:
        """Create scalping trade setup.

        Tight SL, quick target. Slightly larger position size
        since holding period is very short.
        """
        entry_price = contract.ltp
        if entry_price <= 0:
            return None

        if entry_price < self.min_premium or entry_price > self.max_premium:
            return None

        stop_loss = round(entry_price * (1 - self.stop_loss_pct / 100), 2)
        stop_loss = max(stop_loss, 0.05)
        target = round(entry_price * (1 + self.target_pct / 100), 2)

        risk_per_lot = (entry_price - stop_loss) * contract.lot_size
        if risk_per_lot <= 0:
            return None

        # Slightly higher risk for scalping (2.5%) since tight SL
        risk_amount = capital * 0.025
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
                f"Scalp {signal.direction.value} score={signal.score:.0f} "
                f"tech={signal.technical_score:.0f}"
            ),
        )

    def should_exit(
        self,
        position: Position,
        current_price: float,
        spot_price: float,
        current_time: datetime,
    ) -> ExitSignal:
        """Aggressive exit for scalping — never hold too long."""
        if current_time.time() >= self.hard_cutoff:
            return ExitSignal(True, "HARD_CUTOFF", current_price)

        if current_price <= position.stop_loss:
            return ExitSignal(True, "SL_HIT", current_price)

        if current_price >= position.target:
            return ExitSignal(True, "TARGET_HIT", current_price)

        # Strict time limit — scalps should work fast or not at all
        holding_mins = position.holding_duration_minutes
        if holding_mins > self.max_hold_minutes:
            pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
            if pnl_pct < 5:
                # Not enough profit after max hold — exit
                return ExitSignal(
                    True,
                    f"SCALP_TIMEOUT ({holding_mins:.0f}min, P&L={pnl_pct:.1f}%)",
                    current_price,
                )

        # Aggressive trailing (activate early, lock more)
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
        """Always current week (highest gamma for quick moves)."""
        if symbol in INDEX_LOT_SIZES:
            next_exp = get_next_expiry(symbol, current_date)
            dte = days_to_expiry(next_exp, current_date)
            if dte >= 1:
                return next_exp
            return get_next_expiry(symbol, next_exp + timedelta(days=1))
        return get_next_expiry(symbol, current_date)
