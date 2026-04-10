"""NIFTY PCR Extreme Reversal option buying strategy.

When NIFTY's put-call ratio hits extreme levels (>1.3 or <0.7),
it signals contrarian opportunity. Excessive put writing creates a
support floor; excessive call writing creates a resistance ceiling.

NIFTY-specific: NIFTY's massive OI (50+ lakh crore notional) makes
PCR readings extremely meaningful — genuine institutional positioning,
not retail noise.

Best for: Range-bound markets (Tue-Wed), when PCR has been at extremes
for 2-3 sessions.
"""

from datetime import date, datetime, time, timedelta
from typing import Optional

from src.broker.models import OptionContract, Position
from src.signals.models import Direction, Signal
from src.strategies.base import BaseStrategy, ExitSignal, TradeSetup
from src.utils.helpers import get_next_expiry, days_to_expiry
from src.utils.logger import get_logger

logger = get_logger("nifty_pcr_reversal")


class NiftyPCRReversalStrategy(BaseStrategy):
    """Buy NIFTY options on extreme PCR mean reversion."""

    name = "nifty_pcr_reversal"

    def __init__(
        self,
        min_score: float = 68.0,
        target_pct: float = 40.0,
        stop_loss_pct: float = 25.0,
        trailing_activation_pct: float = 20.0,
        trailing_lock_pct: float = 50.0,
        time_exit_minutes: int = 90,
        hard_cutoff: time = time(15, 0),
        entry_start: time = time(9, 45),
        entry_end: time = time(13, 30),
        bullish_pcr_threshold: float = 1.3,
        bearish_pcr_threshold: float = 0.7,
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
        self.bullish_pcr_threshold = bullish_pcr_threshold
        self.bearish_pcr_threshold = bearish_pcr_threshold

        # PCR state — set externally by the engine
        self.current_pcr: float = 0.0
        self.max_put_oi_strike: int = 0
        self.max_call_oi_strike: int = 0

    def set_pcr_data(
        self,
        pcr: float,
        max_put_oi_strike: int = 0,
        max_call_oi_strike: int = 0,
    ):
        """Update current PCR and OI wall data."""
        self.current_pcr = pcr
        self.max_put_oi_strike = max_put_oi_strike
        self.max_call_oi_strike = max_call_oi_strike

    def should_enter(
        self,
        signal: Signal,
        spot_price: float,
        current_time: datetime,
    ) -> bool:
        """Enter when PCR is at extreme + signal confirms reversal.

        Bullish: PCR > 1.3 (put writers defend support) + bullish signal
        Bearish: PCR < 0.7 (call writers cap resistance) + bearish signal
        """
        if signal.symbol != "NIFTY":
            return False

        if signal.score < self.min_score:
            return False

        if current_time.time() < self.entry_start:
            return False

        if current_time.time() >= self.entry_end:
            return False

        if self.current_pcr <= 0:
            return False

        # Determine PCR-implied direction
        pcr_direction = Direction.NEUTRAL
        if self.current_pcr >= self.bullish_pcr_threshold:
            pcr_direction = Direction.BULLISH
        elif self.current_pcr <= self.bearish_pcr_threshold:
            pcr_direction = Direction.BEARISH

        if pcr_direction == Direction.NEUTRAL:
            return False

        # Signal must agree with PCR reversal direction
        if signal.direction != pcr_direction:
            return False

        # OI direction should support or be neutral
        if signal.oi_direction != Direction.NEUTRAL and signal.oi_direction != pcr_direction:
            return False

        # Proximity check: for bullish, spot should be near put-OI support
        if pcr_direction == Direction.BULLISH and self.max_put_oi_strike > 0:
            distance = spot_price - self.max_put_oi_strike
            if distance > 200:  # Too far above support
                return False

        if pcr_direction == Direction.BEARISH and self.max_call_oi_strike > 0:
            distance = self.max_call_oi_strike - spot_price
            if distance > 200:  # Too far below resistance
                return False

        logger.info(
            f"PCR REVERSAL: NIFTY {pcr_direction.value} PCR={self.current_pcr:.2f} "
            f"spot={spot_price:.0f} put_wall={self.max_put_oi_strike} "
            f"call_wall={self.max_call_oi_strike}"
        )
        return True

    def create_setup(
        self,
        signal: Signal,
        contract: OptionContract,
        spot_price: float,
        capital: float,
    ) -> Optional[TradeSetup]:
        """Create PCR reversal setup — contrarian sizing."""
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

        # 1.5% risk — contrarian play
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
                f"PCR Reversal {signal.direction.value} "
                f"PCR={self.current_pcr:.2f} score={signal.score:.0f}"
            ),
        )

    def should_exit(
        self,
        position: Position,
        current_price: float,
        spot_price: float,
        current_time: datetime,
    ) -> ExitSignal:
        """Exit with time limit — PCR reversals play out fast."""
        if current_time.time() >= self.hard_cutoff:
            return ExitSignal(True, "HARD_CUTOFF", current_price)

        if current_price <= position.stop_loss:
            return ExitSignal(True, "SL_HIT", current_price)

        if current_price >= position.target:
            return ExitSignal(True, "TARGET_HIT", current_price)

        # Strict 90 min time limit
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
        """Next week — reversals need time to unfold."""
        next_exp = get_next_expiry(symbol, current_date)
        dte = days_to_expiry(next_exp, current_date)
        if dte >= 3:
            return next_exp
        return get_next_expiry(symbol, next_exp + timedelta(days=1))
