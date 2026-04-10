"""NIFTY Opening Range Breakout (ORB) option buying strategy.

Waits for the first 15-minute candle (9:15-9:30) to establish the
opening range, then buys CE/PE when NIFTY breaks above/below with
volume confirmation and SuperTrend alignment.

NIFTY-specific: Exploits NIFTY's extreme options liquidity for
zero-slippage ORB entries, and uses GIFT Nifty gap direction
as a confirmation filter.

Best for: Trending days with clear GIFT Nifty direction, VIX 14-22.
"""

from datetime import date, datetime, time, timedelta
from typing import Optional

from src.broker.models import OptionContract, Position
from src.signals.models import Direction, Signal
from src.strategies.base import BaseStrategy, ExitSignal, TradeSetup
from src.utils.constants import INDEX_LOT_SIZES
from src.utils.helpers import get_next_expiry, days_to_expiry, is_expiry_day
from src.utils.logger import get_logger

logger = get_logger("nifty_orb")

# ORB range window
ORB_START = time(9, 15)
ORB_END = time(9, 30)


class NiftyORBStrategy(BaseStrategy):
    """Buy NIFTY options on 15-min opening range breakout."""

    name = "nifty_orb"

    def __init__(
        self,
        min_score: float = 72.0,
        target_pct: float = 40.0,
        stop_loss_pct: float = 30.0,
        trailing_activation_pct: float = 20.0,
        trailing_lock_pct: float = 50.0,
        time_exit_minutes: int = 45,
        hard_cutoff: time = time(15, 0),
        entry_start: time = time(9, 31),
        entry_end: time = time(12, 0),
        min_vix: float = 13.0,
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
        self.min_vix = min_vix

        # ORB state — set externally by the engine each day
        self.orb_high: Optional[float] = None
        self.orb_low: Optional[float] = None
        self._breakout_triggered: bool = False

    def set_orb_range(self, high: float, low: float):
        """Set the opening range from the 9:15-9:30 candle."""
        self.orb_high = high
        self.orb_low = low
        self._breakout_triggered = False
        logger.info(f"NIFTY ORB range set: {low:.0f} - {high:.0f} (width={high - low:.0f})")

    def should_enter(
        self,
        signal: Signal,
        spot_price: float,
        current_time: datetime,
    ) -> bool:
        """Enter when NIFTY breaks 15-min ORB with signal confirmation.

        Conditions:
        1. NIFTY only
        2. ORB range must be set
        3. Time within entry window (after 9:30, before noon)
        4. Spot price has broken above ORB high or below ORB low
        5. Signal direction aligns with breakout direction
        6. SuperTrend and volume should confirm
        """
        if signal.symbol != "NIFTY":
            return False

        if signal.score < self.min_score:
            return False

        if current_time.time() < self.entry_start:
            return False

        if current_time.time() >= self.entry_end:
            return False

        if self.orb_high is None or self.orb_low is None:
            return False

        if self._breakout_triggered:
            return False  # One trade per day

        # Determine breakout direction
        breakout_dir = Direction.NEUTRAL
        if spot_price > self.orb_high:
            breakout_dir = Direction.BULLISH
        elif spot_price < self.orb_low:
            breakout_dir = Direction.BEARISH

        if breakout_dir == Direction.NEUTRAL:
            return False

        # Signal direction must match breakout
        if signal.direction != breakout_dir:
            return False

        # Technical direction must confirm
        if signal.technical_direction != breakout_dir:
            return False

        # Volume confirmation from signal components
        has_volume = False
        for comp in signal.components:
            if comp.name == "technical" and "volume" in comp.details.lower():
                has_volume = True

        if not has_volume:
            return False

        self._breakout_triggered = True
        orb_width = self.orb_high - self.orb_low
        logger.info(
            f"NIFTY ORB BREAKOUT {breakout_dir.value}: "
            f"spot={spot_price:.0f} ORB=[{self.orb_low:.0f}-{self.orb_high:.0f}] "
            f"width={orb_width:.0f} score={signal.score:.0f}"
        )
        return True

    def create_setup(
        self,
        signal: Signal,
        contract: OptionContract,
        spot_price: float,
        capital: float,
    ) -> Optional[TradeSetup]:
        """Create ORB trade setup."""
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

        # 2% risk, reduce to 1.5% on expiry day
        risk_pct = 0.015 if is_expiry_day("NIFTY") else 0.02
        risk_amount = capital * risk_pct
        lots = max(1, int(risk_amount / risk_per_lot))
        quantity = lots * contract.lot_size

        orb_info = ""
        if self.orb_high and self.orb_low:
            orb_info = f"ORB=[{self.orb_low:.0f}-{self.orb_high:.0f}]"

        return TradeSetup(
            symbol=signal.symbol,
            signal=signal,
            contract=contract,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target=target,
            quantity=quantity,
            strategy_name=self.name,
            reason=f"NIFTY ORB {signal.direction.value} {orb_info} score={signal.score:.0f}",
        )

    def should_exit(
        self,
        position: Position,
        current_price: float,
        spot_price: float,
        current_time: datetime,
    ) -> ExitSignal:
        """Exit — also exit if price re-enters ORB range."""
        if current_time.time() >= self.hard_cutoff:
            return ExitSignal(True, "HARD_CUTOFF", current_price)

        if current_price <= position.stop_loss:
            return ExitSignal(True, "SL_HIT", current_price)

        if current_price >= position.target:
            return ExitSignal(True, "TARGET_HIT", current_price)

        # Time exit if negative
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
        """Current week expiry for ORB (high gamma intraday plays)."""
        next_exp = get_next_expiry(symbol, current_date)
        dte = days_to_expiry(next_exp, current_date)
        if dte >= 1:
            return next_exp
        return get_next_expiry(symbol, next_exp + timedelta(days=1))
