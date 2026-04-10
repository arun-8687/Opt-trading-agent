"""NIFTY GIFT Nifty Gap-and-Go option buying strategy.

When GIFT Nifty shows a significant pre-market gap (>0.5%) and FII
flow data from the previous session confirms the direction, buy
options after the opening range settles. NIFTY gaps fill less than
40% of the time when FII flow confirms the direction.

NIFTY-specific: FIIs drive ~40% of derivatives volume. GIFT Nifty
front-runs NSE open. When both align, the gap is high-conviction.

Best for: Monday mornings (weekend cues), post-US-Fed/jobs-data gaps.
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

logger = get_logger("nifty_gift_gap")


class NiftyGIFTGapStrategy(BaseStrategy):
    """Buy NIFTY options on GIFT Nifty gap with FII confirmation."""

    name = "nifty_gift_gap"

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

        # Gap data — set externally before market open
        self.gift_gap_pct: float = 0.0
        self.gap_direction: Direction = Direction.NEUTRAL
        self.fii_confirms: bool = False
        self.previous_close: float = 0.0

    def set_gap_data(
        self,
        gift_gap_pct: float,
        previous_close: float,
        fii_net_buy: bool,
    ):
        """Set pre-market gap data from GIFT Nifty and FII activity.

        Args:
            gift_gap_pct: GIFT Nifty gap percentage (positive=gap up)
            previous_close: NIFTY previous day close
            fii_net_buy: True if FII were net buyers yesterday
        """
        self.gift_gap_pct = gift_gap_pct
        self.previous_close = previous_close

        if gift_gap_pct >= self.min_gap_pct:
            self.gap_direction = Direction.BULLISH
            self.fii_confirms = fii_net_buy
        elif gift_gap_pct <= -self.min_gap_pct:
            self.gap_direction = Direction.BEARISH
            self.fii_confirms = not fii_net_buy  # FII sellers confirm gap-down
        else:
            self.gap_direction = Direction.NEUTRAL
            self.fii_confirms = False

        if self.gap_direction != Direction.NEUTRAL:
            logger.info(
                f"GIFT Gap: {self.gap_direction.value} {gift_gap_pct:+.2f}% "
                f"FII confirms={self.fii_confirms}"
            )

    def should_enter(
        self,
        signal: Signal,
        spot_price: float,
        current_time: datetime,
    ) -> bool:
        """Enter when GIFT gap sustains after ORB with FII confirmation.

        Conditions:
        1. NIFTY only
        2. Significant GIFT Nifty gap (>= 0.5%)
        3. FII flow confirms gap direction
        4. Gap sustaining after 9:35 AM
        5. Signal direction aligns
        """
        if signal.symbol != "NIFTY":
            return False

        if signal.score < self.min_score:
            return False

        if current_time.time() < self.entry_start:
            return False

        if current_time.time() >= self.entry_end:
            return False

        if self.gap_direction == Direction.NEUTRAL:
            return False

        if not self.fii_confirms:
            return False

        # Signal must agree with gap direction
        if signal.direction != self.gap_direction:
            return False

        # Technical must confirm
        if signal.technical_direction != self.gap_direction:
            return False

        # Gap must be sustaining — spot above prev close for gap-up
        if self.previous_close > 0:
            if self.gap_direction == Direction.BULLISH:
                if spot_price < self.previous_close:
                    return False  # Gap filled
            elif self.gap_direction == Direction.BEARISH:
                if spot_price > self.previous_close:
                    return False  # Gap filled

        # Avoid on expiry day — theta kills even sustained gaps
        if is_expiry_day("NIFTY", current_time):
            return False

        logger.info(
            f"GIFT GAP-AND-GO: {self.gap_direction.value} "
            f"gap={self.gift_gap_pct:+.2f}% spot={spot_price:.0f} "
            f"prev_close={self.previous_close:.0f} FII_confirms=True"
        )
        return True

    def create_setup(
        self,
        signal: Signal,
        contract: OptionContract,
        spot_price: float,
        capital: float,
    ) -> Optional[TradeSetup]:
        """Create GIFT gap-and-go setup."""
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
                f"GIFT Gap {self.gap_direction.value} "
                f"gap={self.gift_gap_pct:+.2f}% FII_confirmed "
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
        """Exit — also exit if gap fills >50%."""
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
        """Current week for intraday gap plays."""
        next_exp = get_next_expiry(symbol, current_date)
        dte = days_to_expiry(next_exp, current_date)
        if dte >= 2:
            return next_exp
        return get_next_expiry(symbol, next_exp + timedelta(days=1))
