"""NIFTY Expiry Day Gamma Blast option buying strategy.

On weekly Thursday expiry, ATM options have extreme gamma and near-zero
time value. A 50-100 point NIFTY move causes 2-5x premium multiplier.
Waits for tight consolidation then buys the breakout direction.

NIFTY-specific: Exploits the 50-point strike interval for maximum gamma
and the massive expiry-day volume for liquidity.

Best for: Expiry Thursdays with tight morning consolidation.
"""

from datetime import date, datetime, time, timedelta
from typing import Optional

from src.broker.models import OptionContract, Position
from src.signals.models import Direction, Signal
from src.strategies.base import BaseStrategy, ExitSignal, TradeSetup
from src.utils.helpers import (
    days_to_expiry,
    get_next_expiry,
    is_expiry_day,
)
from src.utils.logger import get_logger

logger = get_logger("nifty_gamma_blast")


class NiftyGammaBlastStrategy(BaseStrategy):
    """Buy cheap ATM options on NIFTY expiry day range breakout."""

    name = "nifty_gamma_blast"

    def __init__(
        self,
        min_score: float = 65.0,
        target_pct: float = 100.0,
        stop_loss_pct: float = 40.0,
        trailing_activation_pct: float = 50.0,
        trailing_lock_pct: float = 60.0,
        hard_cutoff: time = time(15, 15),
        entry_start: time = time(13, 45),
        entry_end: time = time(15, 0),
        max_consolidation_range: float = 50.0,
        min_premium: float = 2.0,
        max_premium: float = 60.0,
    ):
        self.min_score = min_score
        self.target_pct = target_pct
        self.stop_loss_pct = stop_loss_pct
        self.trailing_activation_pct = trailing_activation_pct
        self.trailing_lock_pct = trailing_lock_pct
        self.hard_cutoff = hard_cutoff
        self.entry_start = entry_start
        self.entry_end = entry_end
        self.max_consolidation_range = max_consolidation_range
        self.min_premium = min_premium
        self.max_premium = max_premium

        # Day's consolidation range — set externally
        self.day_high: Optional[float] = None
        self.day_low: Optional[float] = None

    def set_day_range(self, high: float, low: float):
        """Set the current day's high/low for consolidation check."""
        self.day_high = high
        self.day_low = low

    def should_enter(
        self,
        signal: Signal,
        spot_price: float,
        current_time: datetime,
    ) -> bool:
        """Enter only on NIFTY expiry day after afternoon breakout.

        Conditions:
        1. NIFTY only, on expiry day (Thursday)
        2. Day's range < 50 points (tight consolidation)
        3. After 1:45 PM, breakout above day high or below day low
        4. Signal direction aligns with breakout
        """
        if signal.symbol != "NIFTY":
            return False

        if not is_expiry_day("NIFTY", current_time):
            return False

        if signal.score < self.min_score:
            return False

        if current_time.time() < self.entry_start:
            return False

        if current_time.time() >= self.entry_end:
            return False

        if self.day_high is None or self.day_low is None:
            return False

        # Check consolidation — day range should be tight
        day_range = self.day_high - self.day_low
        if day_range > self.max_consolidation_range:
            return False

        # Check breakout
        breakout_dir = Direction.NEUTRAL
        if spot_price > self.day_high:
            breakout_dir = Direction.BULLISH
        elif spot_price < self.day_low:
            breakout_dir = Direction.BEARISH

        if breakout_dir == Direction.NEUTRAL:
            return False

        if signal.direction != breakout_dir:
            return False

        logger.info(
            f"GAMMA BLAST: NIFTY {breakout_dir.value} breakout on expiry | "
            f"range={day_range:.0f}pts spot={spot_price:.0f} "
            f"day=[{self.day_low:.0f}-{self.day_high:.0f}]"
        )
        return True

    def create_setup(
        self,
        signal: Signal,
        contract: OptionContract,
        spot_price: float,
        capital: float,
    ) -> Optional[TradeSetup]:
        """Create gamma blast setup with cheap options."""
        entry_price = contract.ltp
        if entry_price <= 0:
            return None

        # Only trade cheap options for gamma explosion
        if entry_price < self.min_premium or entry_price > self.max_premium:
            return None

        stop_loss = round(entry_price * (1 - self.stop_loss_pct / 100), 2)
        stop_loss = max(stop_loss, 0.05)
        target = round(entry_price * (1 + self.target_pct / 100), 2)

        risk_per_lot = (entry_price - stop_loss) * contract.lot_size
        if risk_per_lot <= 0:
            return None

        # Conservative 1.5% risk — binary outcome expected
        risk_amount = capital * 0.015
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
                f"Gamma Blast {signal.direction.value} "
                f"premium={entry_price:.1f} "
                f"range=[{self.day_low:.0f}-{self.day_high:.0f}]"
            ),
        )

    def should_exit(
        self,
        position: Position,
        current_price: float,
        spot_price: float,
        current_time: datetime,
    ) -> ExitSignal:
        """Aggressive exit — must exit before 3:15 PM on expiry."""
        if current_time.time() >= self.hard_cutoff:
            return ExitSignal(True, "EXPIRY_CUTOFF", current_price)

        if current_price <= position.stop_loss:
            return ExitSignal(True, "SL_HIT", current_price)

        if current_price >= position.target:
            return ExitSignal(True, "TARGET_HIT", current_price)

        # Near-zero exit — option about to expire worthless
        if current_price < 1.0 and position.entry_price > 3.0:
            return ExitSignal(True, "NEAR_ZERO", current_price)

        # Trailing SL — once option doubles, move SL to cost
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
        """Current week expiry (we only trade on expiry day itself)."""
        next_exp = get_next_expiry(symbol, current_date)
        dte = days_to_expiry(next_exp, current_date)
        if dte >= 0:
            return next_exp
        return get_next_expiry(symbol, next_exp + timedelta(days=1))
